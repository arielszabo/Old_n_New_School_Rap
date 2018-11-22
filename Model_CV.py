import pandas as pd
import numpy as np
import os
import re
import sys
import logging
import spacy
import datetime
from sklearn.model_selection import RepeatedStratifiedKFold
from dask_searchcv import GridSearchCV
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from ariels_utils import MLTester
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import metrics
import xgboost as xgb
import hashlib
import glob


def logging_setup(output_folder, output_file_name):
    file_dirname = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(file_dirname, output_folder)
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s',
                        filename=os.path.join(output_path, output_file_name))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
    logging.getLogger('').addHandler(ch)

    logger = logging.getLogger(__name__)
    return logger


def load_exported_file(input_data_path):
    input_mtime = str(os.path.getmtime(input_data_path)).replace('.', '')
    files_found = glob.glob(os.path.join('code_output', f'*{input_mtime}*'), recursive=True)
    if len(files_found) > 1:
        logging.warning(f"There is more than 1 files:\n {files_found}")
    elif len(files_found) == 1:
        logging.info(f"Found a good file so there is no need to compute BOW again:\n {files_found[0]}")
        return pd.read_csv(os.path.join('code_output', files_found[0]))
    else:
        return None


class CustomVectorizer(CountVectorizer):
    def __init__(self, language='xx', *args, **kwargs):
        super(CustomVectorizer, self).__init__(*args, **kwargs)
        self.language = language
        self.nlp = spacy.load(self.language)  # todo: if it's not english ?

    # create the analyzer that will be returned by this method
    def custom_analyzer(self, doc):
        # apply the preprocessing and tokenzation steps

        tokens = self.nlp(doc.lower())
        lemmatized_tokens = [token.lemma_ for token in tokens]

        # use sklearn-vectorizer's _word_ngrams built in method
        # to remove stop words and extract n-grams
        return self._word_ngrams(lemmatized_tokens, self.get_stop_words())

    # overwrite the build_analyzer method, allowing one to
    # create a custom analyzer for the vectorizer
    def build_analyzer(self):
        return self.custom_analyzer


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
    logging.info('Clean bad input lyrics')
    data['word_count'] = data['lyrics'].apply(lambda lyrics: len(lyrics.split()))
    # Drop high word count
    data.drop(data.query('word_count > 1300').index, inplace=True)
    logging.info('Drop high word count')
    # Drop low word count
    data.drop(data.query('word_count < 50').index, inplace=True)
    logging.info('Drop low word count')
    # Drop duplicates
    data.drop_duplicates(subset=['lyrics'], inplace=True)
    logging.info('Drop duplicate lyrics')
    data['clean_lyrics'] = data['lyrics'].apply(clean_song_lyrics)
    logging.info('clean lyrics')

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
    logging.info('clean slang word convertor')

    # custom clean names
    custom_names = set(data.artist)
    custom_names.update(['bad boy', 'queen bee'])
    for artist_name in custom_names:
        data['clean_lyrics'].str.replace(f'[\b\s]({artist_name.lower()})[\b\s]', '')
    logging.info('clean custom names from lyrics')

    return data


if __name__ == '__main__':
    logging_setup('code_output', f'rap_model_CV_{str(datetime.datetime.now().date())}.log')
    input_data = r'lyrics2.xlsx'
    df = pd.read_excel(input_data)
    clean_data = clean_df(df)

    pre_model = CustomVectorizer(stop_words='english', ngram_range=(1, 2), language='en', binary=True)

    model = Pipeline(
        [
            ('tfidf', TfidfTransformer()),
            ('xgb', xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                                      objective='binary:logistic', booster='gblinear', n_jobs=-1, nthread=None,
                                      gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                                      colsample_bytree=1,
                                      colsample_bylevel=1, reg_alpha=0, reg_lambda=1))
            # ('svm', SVC(gamma=3, C=100))
        ]
    )

    X = clean_data['clean_lyrics']
    y = clean_data['rapper_type']

    logging.info('Starting_to_train')
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # The next chapter is for run time saving, there is no need to do all the CustomVectorizer
    # if we already did it once and the data haven't changed:

    vec_train = load_exported_file(input_data)
    # There isn't a file with the same name as the last modified date of the input data (aka data has changed):
    if not vec_train:
        vec_train_arr = pre_model.fit_transform(x_train)
        np.save(vec_train_arr, f'vec_train_{str(os.path.getmtime(input_data)).replace(".", "")}.npy')
        # vec_feature_names = pre_model.get_feature_names()
        # vec_train = pd.DataFrame(vec_train_arr.toarray(), columns=pre_model.get_feature_names())
        # vec_train.to_csv(f'vec_train_{str(os.path.getmtime(input_data)).replace(".", "")}.csv', chunksize=100)

    # Fit the rest of the model

    model.fit(vec_train, y_train)
    y_pred_train = model.predict(vec_train)
    logging.info(metrics.classification_report(y_train, y_pred_train))

    x_test_vec = pre_model.transform(x_test)
    y_pred = model.predict(x_test_vec)
    logging.info(metrics.classification_report(y_test, y_pred))

    exit()
    print(MLTester(model, X, y, scoring_method='f1_weighted', n_jobs=3, splitting_method=RepeatedStratifiedKFold,
                   splitting_method_params={'n_splits': 5}).run())

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

    gcv = GridSearchCV(model, params, cv=4, n_jobs=1, scoring='f1_weighted')
    gcv.fit(X, y)
    print(gcv.best_score_, gcv.best_params_)
