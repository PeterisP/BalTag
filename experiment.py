import os
import time
import datetime
import pickle
from gensim.models import Word2Vec
from collections import namedtuple
from lt_document import LTDocument
from vocabularies import Vocabularies
from feature_factory import FeatureFactory
from tagger import Tagger

# Experiment configuration
Configuration = namedtuple('Configuration', ['input_embeddings', 'input_wordshape', 'input_ngrams', 'small_sample',
                                             'rare_word_limit', 'max_ngrams', 'default_epochs', 'pos_only',
                                             'convolution_layer', 'full_layer', 'rnn_layers', 'wide_and_deep',
                                             'compressed_width', 'convolution_width', 'full_layer_width', 'rnn_width'])

EXPERIMENT_NR = 10
configuration = Configuration(
    small_sample=False,  # Iff true, use only a small subsample for a proof of concept test
    rare_word_limit=10,  # Only words and n-grams seen at least this many times will be used as features
    input_embeddings=True,
    input_wordshape=True,
    input_ngrams=True,
    max_ngrams=4,
    default_epochs=5,
    pos_only=False,
    convolution_layer=False,
    full_layer=True,
    rnn_layers=1,
    wide_and_deep=True,
    compressed_width=400,
    convolution_width=500,
    full_layer_width=400,
    rnn_width=200,
)

# Set up of constants
TRAIN_DATA_FILENAME = 'data/train.json'
EVAL_DATA_FILENAME = 'data/dev.json'
EMBEDDINGS_FILENAME = 'data/lt_polyglot_embeddings.pkl'
TRAIN_LIMIT = 3000 if configuration.small_sample else None
EVAL_LIMIT = 1000 if configuration.small_sample else None
OUTPUT_DIR = 'exp{:03}'.format(EXPERIMENT_NR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_embeddings():
    if configuration.input_embeddings:
        with open(EMBEDDINGS_FILENAME, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    else:
        return None

def train_tagger(tagger, epochs=configuration.default_epochs):
    print('Loading...')
    # Loading documents
    train_doc = LTDocument(TRAIN_DATA_FILENAME, TRAIN_LIMIT, configuration.pos_only)
    eval_doc = LTDocument(EVAL_DATA_FILENAME, EVAL_LIMIT, configuration.pos_only)

    # Loading word embeddings
    embeddings = load_embeddings()

    # Build vocabularies and prepare vectorized document data
    vocabularies = Vocabularies(configuration, train_doc)
    featurefactory = FeatureFactory(vocabularies, embeddings)
    train_data = featurefactory.vectorize(train_doc, configuration)
    eval_data = featurefactory.vectorize(eval_doc, configuration)

    if not tagger:
        tagger = Tagger(configuration, featurefactory)
    # Train the model
    print('Training...')
    start_time = time.time()
    tagger.train(train_data, epochs, OUTPUT_DIR, eval_data)
    end_time = time.time()
    print('Trained in {}'.format(datetime.timedelta(seconds=int(end_time - start_time))))
    tagger.dump(OUTPUT_DIR)
    print('Model saved')
    tagger.tag(eval_doc, eval_data, vocabularies, OUTPUT_DIR + '/eval.tagged.txt')
    return tagger


def load_tagger():
    print('Loading tagger ...', end='')
    # Load embeddings
    embeddings = load_embeddings()

    # Load vocabularies
    vocabularies = Vocabularies(configuration, folder=OUTPUT_DIR)
    featurefactory = FeatureFactory(vocabularies, embeddings)

    # Ielādējam tageri
    tagger = Tagger(configuration, featurefactory)
    tagger.load_model(OUTPUT_DIR)
    print('OK')
    return vocabularies, featurefactory, tagger

def train_some_more():
    vocabularies, featurefactory, trained_tagger = load_tagger()

    # Load the documents
    print('Loading documents ...', end='')
    train_doc = LTDocument(TRAIN_DATA_FILENAME, TRAIN_LIMIT, configuration.pos_only)
    train_data = featurefactory.vectorize(train_doc, configuration)
    eval_doc = LTDocument(EVAL_DATA_FILENAME, EVAL_LIMIT, configuration.pos_only)
    eval_data = featurefactory.vectorize(eval_doc, configuration)
    print('OK')

    # Train the model some more
    print('Training...')
    start_time = time.time()
    trained_tagger.train(train_data, 4, OUTPUT_DIR, eval_data)
    end_time = time.time()
    print('Trained in {}'.format(datetime.timedelta(seconds=int(end_time - start_time))))
    trained_tagger.dump(OUTPUT_DIR)
    print('Model saved')

    trained_tagger.tag(eval_doc, eval_data, vocabularies, OUTPUT_DIR + '/eval.tagged.txt')

# trained_tagger = train_tagger(None)
train_some_more()

print('Done!')
