import os
from gensim.models import Word2Vec

from lt_document import LTDocument
from vocabularies import Vocabularies
from feature_factory import FeatureFactory
from tagger import Tagger

# Experiment configuration
EXPERIMENT_NR = 1
SMALL = True  # Iff true, use only a small subsample for a proof of concept test
DEFAULT_EPOCHS = 3
USE_WORDFORM_EMBEDDINGS = False
USE_WORDSHAPE = True
USE_NGRAMS = True
OUTPUT_ATTRIBUTES = True

# Set up of constants
TRAIN_DATA_FILENAME = 'data/train.json'
EVAL_DATA_FILENAME = 'data/dev.json'
EMBEDDINGS_FILENAME = None
SMALL_TRAIN_LIMIT = 1000 if SMALL else None
SMALL_EVAL_LIMIT = 100 if SMALL else None
# TODO - descriptive experiment IDs that include configuration
OUTPUT_DIR = 'exp{:03}'.format(EXPERIMENT_NR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading...')
# Loading documents
train_doc = LTDocument(TRAIN_DATA_FILENAME, SMALL_TRAIN_LIMIT)
eval_doc = LTDocument(EVAL_DATA_FILENAME, SMALL_EVAL_LIMIT)

# Loading word embeddings
if USE_WORDFORM_EMBEDDINGS:
    embeddings = Word2Vec.load_word2vec_format(EMBEDDINGS_FILENAME, binary=True)
else:
    embeddings = None

# Build vocabularies and prepare vectorized document data
vocabularies = Vocabularies(train_doc)
featurefactory = FeatureFactory(vocabularies, embeddings)
train_data = featurefactory.vectorize(train_doc, USE_WORDFORM_EMBEDDINGS, USE_WORDSHAPE, USE_NGRAMS, OUTPUT_ATTRIBUTES)
eval_data = featurefactory.vectorize(eval_doc, USE_WORDFORM_EMBEDDINGS, USE_WORDSHAPE, USE_NGRAMS, OUTPUT_ATTRIBUTES)


def train_stuff(tagger, epochs=DEFAULT_EPOCHS):
    # Uztrenējam modeli
    print('Training...')
    tagger.train(train_data, epochs, OUTPUT_DIR, eval_data)
    tagger.dump(OUTPUT_DIR)
    print('Model saved')
    tagger.tag(eval_doc, eval_data, vocabularies, OUTPUT_DIR + '/eval.tagged.txt')
    return tagger

trained_tagger = train_stuff(Tagger(USE_WORDFORM_EMBEDDINGS, USE_WORDSHAPE, USE_NGRAMS, OUTPUT_ATTRIBUTES, featurefactory))

print('Done!')