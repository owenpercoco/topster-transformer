import spotipy
import sys
import time
from typing import Optional, List
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from operator import itemgetter
import difflib

from credentials import spotify_client_id, spotify_client_secret

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

def get_spotify_auth() -> spotipy.Spotify:
    auth = SpotifyOAuth(client_id=spotify_client_id,
                        client_secret=spotify_client_secret,
                        redirect_uri="http://127.0.0.1:8888/callback",
                        scope="playlist-modify-public playlist-modify-private"
    )
    sp = spotipy.Spotify(auth_manager=auth)
    return sp

def confidence(query, album):
    return difflib.SequenceMatcher(None, query, album).ratio()
    
def search(sp: spotipy.Spotify, query: str):
    resp = sp.search(query, type="album", market="ES", limit=8)
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
        return None

def get_album_link(artist: str, album: str, sp: spotipy.Spotify) -> Optional[str]:
    """Search Spotify and pick the best album link from top artist candidates.

    Steps:
    1. Search for the artist and print the top 3 artist results.
    2. For each of those artists, fetch their albums and print them (limited list).
    3. Compare candidate album names to the requested album using a string similarity
       metric and pick the best candidate across all top artists.

    Returns the Spotify album URL or None if no good match is found.
    """

    print(f"Searching for artist: {artist!r}")
    try:
        artist_res = sp.search(q=artist, type="artist", market='ES', limit=5)
    except Exception as e:
        print(f"Artist search failed: {e}")
        return None

    artist_items = artist_res.get("artists", {}).get("items", [])
    if not artist_items:
        print("No artists found")
        return None


    # Compute name-match confidence for each candidate and pick the best
    scored_artists = []
    for a in artist_items:
        name = (a.get("name") or "").strip()
        score = difflib.SequenceMatcher(None, artist.lower().strip(), name.lower()).ratio()
        scored_artists.append((score, a))

    scored_artists.sort(key=lambda t: t[0], reverse=True)
    best_artist_score, best_artist = scored_artists[0]
    print(f"Best-matching artist: {best_artist.get('name')} (id={best_artist.get('id')}) — confidence={best_artist_score:.3f}")

    # List all albums for the selected artist (handle pagination)
    aid = best_artist.get("id")
    albums = []
    try:
        offset = 0
        while True:
            resp = sp.artist_albums(aid, album_type="album", limit=50, offset=offset)
            items = resp.get("items", [])
            if not items:
                break
            albums.extend(items)
            if not resp.get("next"):
                break
            offset += 50
    except Exception as e:
        print(f"Failed to fetch albums for artist {best_artist.get('name')}: {e}")
        return None

    # Deduplicate album names and prepare scoring
    seen = set()
    candidates = []  # (score, album_id, album_name)
    target = album.lower().strip()
    print(f"Found {len(albums)} album entries (including duplicates/editions); listing unique names:")
    for alb in albums:
        name = (alb.get("name") or "").strip()
        lname = name.lower()
        if lname in seen:
            continue
        seen.add(lname)
        print(f" - {name}")
        score = difflib.SequenceMatcher(None, target, lname).ratio()
        candidates.append((score, alb.get("id"), name))

    if not candidates:
        print("No unique albums found for this artist")
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_album_id, best_album_name = candidates[0]
    print(f"\nBest album match for requested title '{album}': {best_album_name} (id={best_album_id}) — confidence={best_score:.3f}")
    return best_album_id


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
    resp = sp.album('spotify:album:'+album_uri)
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

def run(sp: spotipy.Spotify, query:str, pid: str):
    albums = search(sp, query)
    if not albums or albums[0][2] < .7:
        artist, album = normalize_and_split(query)
        album = get_album_link(artist, album, sp)
    else:
        album = albums[0][1]
    print("got album link: ", album)
    tracks = get_track_ids_from_album(album, sp)
    print("got tracks")
    add_tracks_to_playlist(pid, tracks, sp)
    print("added to playlist")

if __name__ == "__main__":

    auth = SpotifyOAuth(client_id=spotify_client_id,
                        client_secret=spotify_client_secret,
                        redirect_uri="http://127.0.0.1:8888/callback",
                        scope="playlist-modify-public playlist-modify-private"
    )
    sp = spotipy.Spotify(auth_manager=auth)
    run(sp, 'Anna Von Hausswolf', 'IconoJast')
    #print(album)
    print('--- End ---')

