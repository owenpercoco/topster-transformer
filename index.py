"""Simple OCR utility using pytesseract to read `chart.webp` in this directory.

Assumes the Python dependencies are installed (no import guards):
- Pillow (PIL)
- pytesseract

Also requires the Tesseract OCR binary to be installed on the system
(e.g. `brew install tesseract` on macOS).

This file provides `extract_and_print_text_from_chart()` which returns the extracted
text and prints it to stdout.
"""

import os
from typing import Optional, List
import easyocr
from PIL import Image, ImageOps, ImageFilter
import numpy as np

from spotify import get_spotify_auth, run, create_playlist


def preprocess_image(path: str, upscale: int = 4) -> Image.Image:
    """Return a preprocessed PIL Image suitable for  OCR.

    Steps: grayscale, autocontrast, mild denoise, optional upscale.
    """
    im = Image.open(path).convert("L")
    im = ImageOps.autocontrast(im)
    im = im.filter(ImageFilter.MedianFilter(size=3))
    if upscale and upscale != 1:
        im = im.resize((im.width * upscale, im.height * upscale), Image.BILINEAR)
    return im



def find_nonblack_block(img: Image.Image, black_threshold: int = 0.5, black_fraction: float = 0.95, starting_point: int = 0) -> Optional[tuple]:
    """Scan image rows from top and return the first non-black block as (start_y, end_y).

    Behavior:
    - Converts the image to grayscale and treats a pixel as black when its value <= black_threshold.
    - Finds the first row index where not all pixels are black => start_y.
    - Continues scanning until it finds a row where all pixels are black again => end_y.
    - If no non-black rows are found, returns None.
    - If non-black runs to the bottom of the image, end_y will be the image height.

    Parameters:
    - img: PIL.Image.Image
    - black_threshold: int grayscale threshold (default 0). Use a small >0 to tolerate near-black.

    Returns:
    - (start_y, end_y) tuple of ints, or None if no non-black block is present.
    """
    # Convert to grayscale

    arr = np.array(img)
    if arr.ndim != 2:
        # Unexpected shape, try to collapse
        arr = arr.reshape((arr.shape[0], -1))

    h = arr.shape[0]
    # For each row compute fraction of black pixels (<= black_threshold).
    # Consider a row "non-black" when fraction_black < black_fraction.
    row_black_fraction = np.mean(arr <= black_threshold, axis=1)
    rows_nonblack = row_black_fraction < float(black_fraction)

    # Clamp starting_point
    if starting_point < 0:
        starting_point = 0
    if starting_point >= h:
        return None

    # find first True at or after starting_point
    true_indices = np.where(rows_nonblack[starting_point:])[0]
    if true_indices.size == 0:
        return None

    start_y = int(starting_point + true_indices[0])

    # find first False after start_y (end of this non-black block)
    false_after = np.where(~rows_nonblack[start_y:])[0]
    if false_after.size == 0:
        end_y = h
    else:
        end_y = int(start_y + false_after[0])

    # Try to find the start of the next non-black block after this end.
    next_true = np.where(rows_nonblack[end_y:])[0]
    if next_true.size > 0:
        next_start = int(end_y + next_true[0])
        # Move the boundary to the midpoint between the two blocks so there's spacing
        mid = int((end_y + next_start) / 2)
        end_y = mid

    # Add a small proportional padding around the block to include nearby faint pixels
    block_height = max(1, end_y - start_y)
    pad = max(2, int(block_height * 0.05))
    start_y = max(0, start_y - pad)
    end_y = min(h, end_y + pad)

    return (start_y, end_y)


def read_image(image: Image.Image, reader=None, languages=("en",), gpu=False, paragraph=True) -> List[str]:
    """Run EasyOCR and return a list of detected lines (strings).

    Accepts a PIL Image or a file path. If `reader` is None a temporary reader is created.
    """
    image = np.array(image)
    results = reader.readtext(image, detail=1, paragraph=paragraph)
    lines = [r[1].strip() for r in results if r[1].strip()]
    return " ".join(lines).strip()


def print_all_nonblack_blocks(image: Image.Image, black_threshold: int = 0, black_fraction: float = 0.95) -> List[Image.Image]:
    """Iterate through the image, crop each non-black vertical block and return them as PIL Images.

    Returns:
    - List[Image.Image]: list of cropped block images in the order found.
    """

    crops: List[Image.Image] = []
    start_search = 0
    idx = 1
    width, h = image.size

    while start_search < h:
        res = find_nonblack_block(image, black_threshold=black_threshold, black_fraction=black_fraction, starting_point=start_search)
        if res is None:
            break
        s, e = res
        print(f"Block {idx}: start={s}, end={e}")

        # Crop in-memory and collect
        cropped = image.crop((0, s, width, e))
        crops.append(cropped)


        # Advance start_search to avoid finding the same block again
        next_start = e
        if next_start <= start_search:
            next_start = start_search + 1
        start_search = next_start
        idx += 1

    if not crops:
        print("No non-black blocks found in image")
    return crops


def normalize_and_split(arr) -> List[str]:
    """Normalize input and split into exactly two parts.

    - If `arr` is a list/tuple with exactly two entries, return the two entries stripped.
    - If `arr` is a list/tuple with one entry (or a plain string), attempt to split that
      single string into two pieces using these rules (in order):
        1. Split on the literal pattern ". 3 F I S" (spaces optional, case-insensitive).
        2. If the text contains exactly two whitespace-separated words, split on the first space.
        3. Try splitting on a dash-like separator ("-", "–", "—").
        4. Fallback: return [original_string, ""] so the result always has length 2.

    Always returns a list of two strings (each stripped).
    """
    import re

    # Normalize list/tuple input
    if isinstance(arr, (list, tuple)):
        if len(arr) == 2:
            return [str(arr[0]).strip(), str(arr[1]).strip()]
        if len(arr) == 1:
            s = str(arr[0] or "").strip()
        else:
            # If more than one entry, join with space and treat as single string
            s = " ".join([str(x) for x in arr]).strip()
    else:
        s = str(arr or "").strip()

    # Pattern: ". 3 F I S" allowing arbitrary spacing and case-insensitive
    for ch in ['.', '3', 'F', 'I', 'S']:
        sep = f" {ch} "
        if sep in s:
            parts = s.split(sep, 1)
            left = parts[0].strip()
            right = parts[1].strip() if len(parts) > 1 else ""
            return [left, right]
    # If the string has exactly two whitespace-separated tokens, split on first space
    tokens = s.split()
    if len(tokens) == 2:
        first, second = tokens[0].strip(), tokens[1].strip()
        return [first, second]

    # Try splitting on a dash-like separator
    dash_parts = re.split(r"\s*[-–—]\s*", s, maxsplit=1)
    if len(dash_parts) == 2 and dash_parts[0] and dash_parts[1]:
        return [dash_parts[0].strip(), dash_parts[1].strip()]

    # Fallback: return original in first slot, empty second slot
    return [s, ""]


# (Removed uniform-chop variant; prefer scanning for all non-black blocks.)


def get_files_in_directory(directory_path):
    """
    Returns a list of all files in the specified directory (non-recursive).
    """
    files = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(entry) # Or full_path for absolute paths
    return files



if __name__ == "__main__":
    # Run the OCR on the local chart.webp by default
    print('Creating EasyOCR reader (may take a moment)')
    reader = easyocr.Reader(['en'])
    sp = get_spotify_auth()
    #print('Running extract_albums_from_image on chart.png')
    img = preprocess_image('chart.png')
    blocks = print_all_nonblack_blocks(img)

    index = 1
    pid = create_playlist("test playlist", sp)
    for block in blocks:
        text = read_image(block, reader)
        print("read text as: ", text)
        try:
            run(sp, text, pid)
        except:
            print("guess it didnt work try the next album")
        index = index+1
    
    #run(sp, text[0], text[-1])
    print('--- End ---')

