import argparse
import funcy
import os
import requests
import bs4

from tqdm import tqdm

END_OF_STROPHE = "\nEND_OF_STROPHE\n"
LETRAS_URL = "https://www.letras.mus.br/{artist}"

def main(
    artist,
    save_to
):
    response = requests.get(LETRAS_URL.format(artist=artist))
    soup = bs4.BeautifulSoup(response.content, "html.parser")

    song_list_table = soup.find_all("div", {"class": "songList-table"})

    all_tracks = []

    for track_list in song_list_table:
        tracks = track_list.find_all("li", {"class": "songList-table-row --song isVisible"})
        for track in tracks:
            track_name = track["data-name"]
            track_url = track["data-shareurl"]

            all_tracks.append({
                "artist": artist,
                "track_name": track_name,
                "track_url": track_url
            })

    artist_name = artist.replace("-", " ").title()

    print(f"""
    Retrieved a total of {len(all_tracks)} {artist_name}'s songs
    """)

    for track in tqdm(all_tracks, desc=f"Downloading lyrics from {artist_name}"):
        track_name = track["track_name"].replace("/", "-")
        track_url = track["track_url"]

        response = requests.get(track_url)
        soup = bs4.BeautifulSoup(response.content, "html.parser")

        lyrics_div = soup.find("div", {"class": "lyric-original"})
        strophes = lyrics_div.find_all("p")

        strophe_lyrics = []
        for strophe in strophes:
            verses = funcy.lfilter(
                lambda elem: isinstance(elem, bs4.element.NavigableString),
                strophe.contents
            )

            strophe_lyrics.append("\n".join(verses))

        lyrics = (END_OF_STROPHE).join(strophe_lyrics)

        with open(os.path.join(save_to, f"{track_name}.txt"), "w") as f:
            f.write(lyrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-to", dest="save_to")
    args = parser.parse_args()

    artist = "taylor-swift"

    main(artist, args.save_to)

