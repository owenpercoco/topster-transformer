import spotipy
import sys
import time
from typing import Optional, List
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from operator import itemgetter
import difflib

from credentials import spotify_client_id, spotify_client_secret

def normalize_and_split(query: str, index: int = 1) -> List[str]:
    """
    Split `query` into exactly two parts.

    Rules (in order):
    - If the special marker (any of ". 3 F I S" as " <char> ") is present, split on the first occurrence.
    - If a dash-like separator is present, split on the first occurrence.
    - Otherwise split on the `index`-th space (1 = first space). If index is out of range it's clamped.
    Always returns a list of two stripped strings.
    """
    s = (query or "").strip()
    if not s:
        return ["", ""]

    # Check for the ". 3 F I S" style marker without regex
    for ch in ['.', '3', 'F', 'I', 'S']:
        sep = f" {ch} "
        if sep in s:
            left, right = s.split(sep, 1)
            return [left.strip(), right.strip()]

    # Check for dash-like separators (prefer spaced dash first)
    for sep in [' - ', ' – ', ' — ', '-', '–', '—']:
        if sep in s:
            parts = s.split(sep, 1)
            left = parts[0].strip()
            right = parts[1].strip() if len(parts) > 1 else ""
            return [left, right]

    # Fallback: split on the Nth space (index), always produce two parts
    tokens = s.split()
    if len(tokens) == 1:
        return [s, ""]

    # clamp index to valid range (1..len(tokens)-1)
    if index < 1:
        index = 1
    if index >= len(tokens):
        index = len(tokens) - 1

    left = " ".join(tokens[:index]).strip()
    right = " ".join(tokens[index:]).strip()
    return [left, right]

def create_playlist(playlist_name: str, sp: spotipy.Spotify, public: bool = True, description: str = "Created by script") -> Optional[str]:
    """Create a new playlist for the current authenticated user and return the playlist id.

    Uses Spotify OAuth to obtain a user token with playlist modification scopes. The
    redirect URI defaults to http://localhost:8888/callback; change if needed.
    """
    try:

        user = sp.current_user()
        user_id = user.get("id")
        if not user_id:
            print("Could not determine current user id")
            return None
        resp = sp.user_playlist_create(user_id, playlist_name, public=public, description=description)
        pid = resp.get("id")
        print(f"Created playlist '{playlist_name}' (id={pid}) for user {user_id}")
        return pid
    except Exception as e:
        print(f"Failed to create playlist: {e}")
        return None

def get_track_ids_from_album(album_uri: str, sp: spotipy.Spotify):
    resp = sp.album(album_uri)
    tracks = resp.get("tracks")
    return [r.get("uri") for r in tracks.get("items", [])]

def add_tracks_to_playlist(playlist_id: str, track_ids: List[str], sp: spotipy.Spotify = None) -> bool:
    """Add all tracks from an album to the given playlist.

    If a Spotipy client `sp` is not provided, this function will obtain an OAuth
    token (same scope as playlist creation) that allows modifying the user's
    playlists.
    Returns True on success, False otherwise.
    """
    try:

        # Add tracks to playlist in batches (max 100 per request)
        added = 0
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            sp.playlist_add_items(playlist_id, batch)
            added += len(batch)
            # small sleep to be polite with API limits
            time.sleep(0.2)

        print(f"Added {added} tracks from album to playlist {playlist_id}")
        return True
    except Exception as e:
        print(f"Failed to add album to playlist: {e}")
        return False

def get_spotify_auth() -> spotipy.Spotify:
    auth = SpotifyOAuth(client_id=spotify_client_id,
                        client_secret=spotify_client_secret,
                        redirect_uri="http://127.0.0.1:8888/callback",
                        scope="playlist-modify-public playlist-modify-private"
    )
    sp = spotipy.Spotify(auth_manager=auth)
    return sp

def calculate_highest_confidence(albums: List[any], query:str):
    returns = []
    for item in albums:
          returns.append((item[0], item[1], confidence(item[0], query)))
    returns.sort(key=itemgetter(2), reverse=True)

    return returns

def confidence(query, album):
    return difflib.SequenceMatcher(None, query.lower().strip(), album.lower().strip()).ratio()
    
def search(sp: spotipy.Spotify, query: str):
    resp = sp.search(query, type="album", market="ES", limit=25, offset=0)
    items = resp.get("albums", {}).get("items")
    if items:
        return_info = []
        for item in items:
            text = item.get("artists")[0].get("name") + " - " + item.get("name")
            if "(Deluxe Edition)" in text:
                 text = text.split("(Deluxe Edition)")[0]
            score = confidence(query, text)
            return_info.append((text, item.get("uri").split(':')[-1], score))
        return sorted(return_info, key=itemgetter(2), reverse=True)
        
    else:
        print("no results for this search: ", query)
        return []


def get_artist_albums(sp: spotipy.Spotify, artist_query: str, artist_limit: int = 5, album_limit: int = 7) -> List[tuple]:
    """Search artists matching `artist_query`, fetch up to `album_limit` albums per artist,
    and return a list of ("Artist - Album", album_uri) tuples.

    Parameters:
    - artist_query: search string for artists
    - sp: authenticated spotipy.Spotify client
    - artist_limit: number of artist search results to consider
    - album_limit: number of albums to collect per artist (first N returned)
    """
    results = []
    try:
        artist_res = sp.search(q=artist_query, type="artist", market="ES", limit=artist_limit)
        artist_items = artist_res.get("artists", {}).get("items", [])
    except Exception as e:
        print(f"Artist search failed for '{artist_query}': {e}")
        return results

    seen = set()
    for a in artist_items:
        aname = (a.get("name") or "").strip()
        aid = a.get("id")
        if not aid:
            continue
        try:
            # fetch albums (limit to album_limit)
            resp = sp.artist_albums(aid, album_type="album", limit=album_limit)
            items = resp.get("items", [])
            for it in items:
                alb_name = (it.get("name") or "").strip()
                key = (aname.lower(), alb_name.lower())
                if key in seen:
                    continue
                seen.add(key)
                uri = it.get("uri") or it.get("id")
                text = f"{aname} - {alb_name}"
                results.append((text, uri))
        except Exception as e:
            print(f"Failed to fetch albums for artist {aname}: {e}")
            continue

    return results

def get_albums(sp: spotipy.Spotify, album_query: str, album_limit: int = 7) -> List[tuple]:
    """Search albums matching `album_query`, fetch up to `album_limit` albums per artist,
    and return a list of ("Artist - Album", album_uri) tuples.

    Parameters:
    - artist_query: search string for artists
    - sp: authenticated spotipy.Spotify client
    - artist_limit: number of artist search results to consider
    - album_limit: number of albums to collect per artist (first N returned)
    """
    results = []
    try:
        album_res = sp.search(q=album_query, type="album", market="ES", limit=album_limit)
        album_items = album_res.get("albums", {}).get("items", [])
    except Exception as e:
        print(f"Artist search failed for '{album_query}': {e}")
        return results

    for a in album_items:
        aname = (a.get("name") or "").strip()
        artist = a.get("artists")[0].get("name")
        uri = a.get("uri")
        text = f"{artist} - {aname}"
        results.append((text, uri))

    return results


def run(sp: spotipy.Spotify, query:str, pid: str = None):
    search_results = search(sp, query)
    seen = set()
    if not search_results or search_results[0][2] < .85:
        artist_albums_all: List[tuple] = []
        albums_all: List[tuple] = []
        seen = set()  # dedupe by lowercased "artist - album" text

        tokens = query.split()
        # try every possible split position between words (1 .. len(tokens)-1)
        max_split = max(1, len(tokens) - 1)
        for idx in range(1, max_split + 1):
            artist_part, album_part = normalize_and_split(query, index=idx)

            # collect artist->albums candidates
            a_albums = get_artist_albums(sp, artist_part)
            for text, uri in a_albums:
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                artist_albums_all.append((text, uri))

            # collect album-search candidates
            b_albums = get_albums(sp, album_part)
            for text, uri in b_albums:
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                albums_all.append((text, uri))
        
        combined_candidates = (search_results or []) + artist_albums_all + albums_all

        if combined_candidates:
            search_results = calculate_highest_confidence(combined_candidates, query)
            
        else:
            print("No candidates found for any split")

    print("Best match:", search_results[0])
    album = search_results[0][1]
    if pid:
        tracks = get_track_ids_from_album(album, sp)
        add_tracks_to_playlist(pid, tracks, sp)

if __name__ == "__main__":
    sp = get_spotify_auth()
    query = "Mednesday eleed;"
    run(sp, query)
