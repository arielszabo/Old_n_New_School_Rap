import requests
from bs4 import BeautifulSoup
import os
import json
import logging


class GeniusScraper(object):
    def __init__(self, GENIUS_API_KEY):
        self.base_url = 'http://api.genius.com'
        self.headers = {'Authorization': f'Bearer {GENIUS_API_KEY}'}

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
        while True:
            page_number = 1
            response = requests.get(f'{self.base_url}/artists/{artist_id}/songs?per_page=50&page={page_number}',
                                    headers=self.headers).json()

            if int(response['meta']['status']) > 400:
                print(json.dumps(response, indent=4))

            elif not response['response']['next_page']: # if it is None
                return songs_links

            else:
                for song in response['response']['songs']:
                    if len(songs_links) == number_of_songs_desired:
                        return songs_links
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
        print(artist_id)
        if artist_id:  # if it is not None
            song_urls = self.get_artist_songs(artist_id, number_of_songs_desired)

            lyrics = [self.extract_lyrics_from_webpage(url) for url in song_urls]
            return lyrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

    genius = GeniusScraper('l57FUc_YCYz4NlE_jotgLPlMFi9lBxUxMYlg39K20JUNcT5TQtU1vXiptDcge96G')
    x = genius.songs_lyrics_by_artist('2pac', 100)
    print(x)
    print(len(x))

    old_school_rappers = ['2pac', 'Jay-Z', 'Eazy-E', 'The Notorious B.I.G.', 'Nas', 'Dr. Dre', 'Ice Cube', 'Snoop Dogg',
                          'Moobb Deep', 'Big L']
    new_school_rappers = []



