import pandas as pd
import numpy as np
import os, re
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.feature_extraction import text
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from ariels_utils import MLTester
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedVectorizer_analyzer(TfidfVectorizer):
    def custom_stop_words(self):
        english_stop_words = set(stopwords.words("english"))

        return english_stop_words

    def build_analyzer(self):
        analyzer = super(StemmedVectorizer_analyzer, self).build_analyzer()
        stemmer = SnowballStemmer("english")
        english_stop_words = self.custom_stop_words()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc) if w not in english_stop_words])


# This function gets a raw lyrics string and returns it cleaned after the steps described:
def clean_song_lyrics(raw_song_lyrics):
    # lower-case all lyrics:
    song_lyrics = str(raw_song_lyrics).lower()
    # Remove text between brackets: (I'm not sure this is a good idea, Let's look at this) #todo: ?
    song_lyrics = re.sub(r'\([^\(\)]+\)', '', song_lyrics)
    # Change "$" in a middle of a word to "s":
    song_lyrics = re.sub(r'\w(\$)\w', 's', song_lyrics)
    # all the I'm combinations we will convert to "I am". It seems "I" is something significant:
    song_lyrics = re.sub(r"[\s\b](i.m)[\s\b]", 'i am', song_lyrics)
    # Remove irrelevant text between square brackets like rappers names and verse number ...:
    song_lyrics = re.sub(r'\[[^\[\]]+\]', '', song_lyrics)
    # Remove anything that is not a word, a number or white-space (like ' and ?)
    song_lyrics = re.sub('[^\w\s]', '', song_lyrics)
    return song_lyrics


def clean_df(data):
    skit_songs = data[data.song_url.str.contains('skit', case=False)]
    all_snippet = data[
        data.song_url.str.contains('Snippet', case=False) | data.lyrics.str.contains('[\*\[]Snippet[\*\]]', case=False)
        ]  # This pipe ("|") is an OR
    no_lyrics_in_url = data[~data.song_url.str.contains('lyrics', case=False, regex=False)]
    no_english_lyrics = data[~data.lyrics.str.contains(r'[A-Za-z]')]

    # Drop all the strange, ood, snippet, skit data etc ...
    bad = pd.concat([skit_songs, all_snippet, no_lyrics_in_url, no_english_lyrics])
    data.drop(set(bad.index), inplace=True)
    # Drop high word count
    data.drop(data.query('word_count > 1300').index, inplace=True)
    # Drop low word count
    data.drop(data.query('word_count < 50').index, inplace=True)
    # Drop duplicates
    data.drop_duplicates(subset=['lyrics'], inplace=True)
    data['clean_lyrics'] = data['lyrics'].apply(clean_song_lyrics)
    slang_word_convertor = {
        "aint": "are not",
        "cause": "because",
        "wanna": "want to",
        "em": "them",
        "yeah": "yes",
        "gotta": "got to",
        "yall": "you all",
        "ya": 'you',
        'ho': 'hoe',
        "gon": "going to",
        "gonna": "going to",
        "lil": "little",
        "fuckin": "fucking",
        "motherfuckin": "motherfucking",
        "gettin": "getting",
        "tryna": "trying to",
        "bout": "about",
        "ima": "i am going to",
        "wit": "with",
        "imma": "i am going to",
        "til": "until",
        "talkin": "talking",
        "gangsta": "gangster",
        "comin": "coming",
        "homie": "home boy",
        "livin": "living",
        "runnin": "running",
        "smokin": "smoking",
        "rollin": "rolling",
        "makin": "making",
        "thang": "thing",
        "ridin": "riding",
        "lookin": "looking",
        "gimme": "give me",
        "takin": "taking",
        "playin": "playing",
        "poppin": "popping",
        "sayin": "saying",
        "thinkin": "thinking",
    }
    for key, value in slang_word_convertor.items():
        data['clean_lyrics'] = data['clean_lyrics'].str.replace(f'[\b\s]({key})[\b\s]', value)

    return data


if __name__ == '__main__':
    clean_data = clean_df(pd.read_excel(r'lyrics2.xlsx'))

    model = Pipeline(
            [
                ('vec', StemmedVectorizer_analyzer(binary=True, max_df=0.4,
                                                  min_df=0.001, ngram_range=(1, 2))),
                # ('clening', FunctionTransformer(clean_song_lyrics, validate=False)),
                # ('vec', text.TfidfVectorizer(binary=True, stop_words=english_stop_words, max_df=0.4,
                #                              min_df=0.001, ngram_range=(1, 2))),
                # ('SVD',  TruncatedSVD(n_components=100, n_iter=7)),
                ('min_max', MaxAbsScaler()),
                ('SGD', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.3,
                                      max_iter=1000, epsilon=0.1, n_jobs=-1,
                                      learning_rate='optimal', eta0=0.0, power_t=0.0,
                                      early_stopping=True, validation_fraction=0.15, n_iter_no_change=5))
                # ('logreg', LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1, max_iter=100))
                # ('svm', SVC(gamma=3, C=100))
            ]
        )

    # custom clean names
    custom_names = set(clean_data.artist)
    custom_names.update(['bad boy', 'queen bee'])
    for artist_name in custom_names:
        clean_data['clean_lyrics'].str.replace(f'[\b\s]({artist_name.lower()})[\b\s]', '')

    X = clean_data['clean_lyrics']
    y = clean_data['rapper_type']

    params = {
        #     'logreg__C':np.arange(0.1, 1.5, 0.1), #np.arange(1, 100, 10),#25 #[0.001, 0.01, 0.1, 1, 10, 100]=>10,
        #     'logreg__penalty':['l1', 'l2'], #l2
        #     'vec__binary':[True, False], #True
        #     'vec__ngram_range':[(1, 1), (1, 2)], #(1,2)
        #     'vec__max_df':np.arange(0.1, 0.6, 0.1),
        #     'vec__min_df':np.arange(0.01, 0.1, 0.05)
        #     'vec__min_df':np.arange(0.001, 0.02, 0.005)
        #     'svm__gamma':[0.01, 0.1, 1, 10],
        #     'svm__C':[0.01, 0.1, 1, 10, 100],
        #     'svm__kernel':['linear', 'rbf', 'sigmoid'],
        #     'SVD__n_components':[10, 100, 300],
        #     'SVD__n_iter':np.arange(5, 9),
        #     'SGD__loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        #     'SGD__penalty':['l1', 'l2', 'elasticnet', 'none'],
        #     'SGD__l1_ratio':np.arange(0.1, 1.1, 0.1),
        #     'SGD__learning_rate':['constant', 'optimal', 'invscaling', 'adaptive'],
        #     'SGD__eta0':[0.01],
        'SGD__alpha': np.arange(0.0001, 0.005, 0.0001)
        #     'SGD__power_t':np.arange(0, 1, 0.1)

    }

    gcv = GridSearchCV(model, params, cv=4, n_jobs=3, verbose=3, scoring='f1_weighted')
    gcv.fit(X, y)
    print(gcv.best_score_, gcv.best_params_)
