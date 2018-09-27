import requests
from bs4 import BeautifulSoup
import os
import sqlalchemy
import json
import logging
import pandas as pd


# todo: don't extract same songs
class GeniusScraper(object):
    def __init__(self, GENIUS_API_KEY):
        self.base_url = 'http://api.genius.com'
        self.headers = {'Authorization': f'Bearer {GENIUS_API_KEY}'}
        self.all_songs_links = {}

    def get_artist_id(self, artist_name):
        response = requests.get(f'{self.base_url}/search?q={artist_name}',
                                headers=self.headers).json()

        if int(response['meta']['status']) > 400:
            print(json.dumps(response, indent=4))

        else:
            for hit in response['response']['hits']:
                if hit['result']['primary_artist']['name'].lower() == artist_name.lower():
                    return hit['result']['primary_artist']['id']

    def get_artist_songs(self, artist_id, number_of_songs_desired):
        songs_links = []
        page_number = 1
        while True:
            response = requests.get(f'{self.base_url}/artists/{artist_id}/songs?per_page=50&page={page_number}',
                                    headers=self.headers).json()

            if int(response['meta']['status']) > 400:
                print(json.dumps(response, indent=4))

            elif not response['response']['next_page']:  # if it is None
                return songs_links

            else:
                for song in response['response']['songs']:
                    if len(songs_links) == number_of_songs_desired:
                        return songs_links
                    if not song['url'] in songs_links:
                        songs_links.append(song['url'])
                page_number += 1

    @staticmethod
    def extract_lyrics_from_webpage(page_url):
        page = requests.get(page_url)

        bs = BeautifulSoup(page.text, 'html.parser')

        if bs:
            lyrics = bs.find('div', {'class': 'lyrics'})
            if lyrics:
                return lyrics.get_text()
            else:
                logging.warning(f'This song has no lyrics? at {page_url}')
        else:
            logging.warning(page_url)

    def songs_lyrics_by_artist(self, artist_name, number_of_songs_desired):
        artist_id = self.get_artist_id(artist_name)
        if artist_id:  # if it is not None
            song_urls = self.get_artist_songs(artist_id, number_of_songs_desired)

            # todo: look at this:
            url_n_lyrics = []
            for url in song_urls:
                if url in self.all_songs_links:
                    lyrics = self.all_songs_links[url]
                else:
                    lyrics = self.extract_lyrics_from_webpage(url)
                    self.all_songs_links[url] = self.extract_lyrics_from_webpage(url)

                url_n_lyrics.append((url, lyrics))
            # lyrics = [(url, self.extract_lyrics_from_webpage(url)) for url in song_urls]
            return url_n_lyrics
        else:
            logging.warning(f"Can't find {artist_name}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

    with open('genius_access_token.txt', 'r') as g:  # This file contains the only the genius_access_token
        genius_access_token = g.read()

    genius = GeniusScraper(genius_access_token)
    rappers = {
        'old_school_rappers': ['2pac', 'Jay-Z', 'Eazy-E', 'The Notorious B.I.G.', 'Nas', 'Ice Cube', 'LL Cool J',
                               'Snoop Dogg', 'Mobb Deep', 'Big L', "Lil' Kim", 'N.W.A', 'Run-D.M.C.', 'Redman',
                               'Beastie Boys', 'Ice-T', 'Nate Dogg', 'Xzibit', 'MC Hammer', 'Big Pun', 'PUFF DADDY',
                               'Big Daddy Kane', 'Wu-Tang Clan', 'Warren G', 'Public Enemy', 'DMX', 'Salt-N-Pepa',
                               'Digital Underground', 'Bone Thugs-N-Harmony'],

        'new_school_rappers': ['Tyler, the Creator', 'Schoolboy Q', 'Travis Scott', 'Big Sean', 'Chance The Rapper',
                               'A$AP Rocky', 'J. Cole', 'Drake', 'Wiz Khalifa', 'Nicki Minaj', 'Kendrick Lamar',
                               'Joey Bada$$', 'Logic', 'Kanye West', 'Joyner Lucas', 'Fetty Wap',
                               'Future', '2 Chainz', 'Lil Uzi Vert', 'Mac Miller', 'Rae Sremmurd', 'Lil Wayne',
                               'Pusha-T', 'Lupe Fiasco', 'The Game', 'LIL PUMP', 'Rick Ross', 'Cardi B'],
    }

    data = []
    for rapper_type in rappers:
        for rapper in rappers[rapper_type]:
            logging.info(f'Start extraction of a 100 songs by {rapper}')
            songs_lyrics = genius.songs_lyrics_by_artist(rapper, 100)
            df = pd.DataFrame(songs_lyrics, columns=['song_url', 'lyrics'])
            df['artist'] = rapper
            df['rapper_type'] = rapper_type
            data.append(df)

    final_data = pd.concat(data, ignore_index=True)
    final_data.dropna(inplace=True)  # some songs don't have lyrics
    print(final_data)
    final_data.to_excel(r'lyrics2.xlsx')
    # engine = sqlalchemy.create_engine('sqlite:///old_and_new_school_rappers.db')
    # final_data.to_sql(r'rap_lyrics', con=engine, if_exists='replace', index=False)





