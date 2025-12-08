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

def save_edited_image(img: Image.Image, out_path: str = "edited_image.png") -> str:
    """
    Save a PIL Image to `out_path` (default: 'edited_image.png') and return the path.

    - Converts image mode if necessary so PNG encoding succeeds.
    - Creates parent directory if it doesn't exist.
    - Raises exceptions on I/O errors so callers can handle/report them.
    """
    # Ensure parent directory exists
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Convert to a PNG-friendly mode if required
    # Keep alpha if present, otherwise use RGB
    try:
        if img.mode not in ("RGB", "RGBA", "L"):
            if "A" in img.getbands():
                img_to_save = img.convert("RGBA")
            else:
                img_to_save = img.convert("RGB")
        else:
            img_to_save = img

        img_to_save.save(out_path, format="PNG")
    except Exception:
        # re-raise so the caller can handle the failure, or swap to a fallback behavior
        raise

    return out_path


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


def crop_and_save_image(img: Image.Image, upper: int, lower: int, index: int, out_dir: str = "./croppedblocks/") -> str:
    """Crop a PIL Image to the vertical bounds [upper:lower] and save as PNG.

    Parameters
    - img: PIL.Image.Image to crop
    - upper: upper y-coordinate (inclusive)
    - lower: lower y-coordinate (exclusive)
    - index: integer used to name the output file: edited_image_{index}.png
    - out_dir: directory to save the file (created if necessary)

    Returns the path to the saved file.
    """
    # Normalize and clamp coordinates
    width, height = img.size
    cropped = img.crop((0, upper - 20, width, lower + 20))

    # Build output path and delegate actual saving to save_edited_image()
    out_path = os.path.join(out_dir, f"edited_image_{index}.png")

    # Use the common save helper which handles mode conversion and directory creation
    return save_edited_image(cropped, out_path=out_path)



def read_image(image_or_path, reader=None, languages=("en",), gpu=False, paragraph=True) -> List[str]:
    """Run EasyOCR and return a list of detected lines (strings).

    Accepts a PIL Image or a file path. If `reader` is None a temporary reader is created.
    """

    input_arg = image_or_path
    results = reader.readtext(input_arg, detail=1, paragraph=paragraph)
    lines = [r[1].strip() for r in results if r[1].strip()]
    return " ".join(lines).strip()


def print_all_nonblack_blocks(image_or_path, black_threshold: int = 0, out_dir: str = './cropped/', save: bool = True) -> List[tuple]:
    """Iterate through the image and print (and optionally save) every non-black vertical block.

    Returns a list of (start_y, end_y) tuples in the order they were found.
    """
    if isinstance(image_or_path, Image.Image):
        img = image_or_path
    else:
        img = Image.open(image_or_path).convert('RGB')

    blocks = []
    start_search = 0
    idx = 1
    h = img.size[1]

    while start_search < h:
        res = find_nonblack_block(img, black_threshold=black_threshold, starting_point=start_search)
        if res is None:
            break
        s, e = res
        print(f"Block {idx}: start={s}, end={e}")
        blocks.append((s, e))
        if save:
            try:
                path = crop_and_save_image(img, s, e, idx, out_dir=out_dir)
                print(f"  Saved block as: {path}")
            except Exception as exc:
                print(f"  Failed to save block {idx}: {exc}")

        # Advance start_search to the row after the end to avoid finding the same block again
        next_start = e
        if next_start <= start_search:
            # Defensive guard to prevent infinite loops
            next_start = start_search + 1
        start_search = next_start
        idx += 1

    if not blocks:
        print("No non-black blocks found in image")
    return blocks


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
    #img = preprocess_image('chart.png')
    
    #blocks = print_all_nonblack_blocks(img)
    files = get_files_in_directory('./cropped')
    index = 1
    pid = create_playlist("test playlist", sp)
    for file in files:
        print(f'./cropped/edited_image_{index}.png')
        text = read_image(f'./cropped/edited_image_{index}.png', reader)
        print("read text as: ", text)
        try:
            run(sp, text, pid)
        except:
            print("guess it didnt work try the next album")
        index = index+1
    
    #run(sp, text[0], text[-1])
    print('--- End ---')

