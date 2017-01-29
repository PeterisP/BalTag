from tensorflow.contrib.learn.python.learn.preprocessing import CategoricalVocabulary
from lt_document import WORDFORM_KEY, ATTRIBUTE_KEY
import pickle

# Configuration - TODO - move to tagger
MAX_NGRAMS = 3  # Length of word suffix letter n-grams to use as features
REMOVE_RARE_WORDS = True
UNK = '_UNK_'


# Management of vocabularies to map words and attributes to numeric identifiers
class Vocabularies(object):
    def __init__(self, document=None, folder=None):
        if folder:
            with open(folder + '/voc_wordforms.p', 'rb') as f:
                self.voc_wordforms = pickle.load(f)
            with open(folder + '/voc_ngrams.p', 'rb') as f:
                self.voc_ngrams = pickle.load(f)
            with open(folder + '/voc_attributes.p', 'rb') as f:
                self.voc_attributes = pickle.load(f)
        else:
            self.voc_wordforms = CategoricalVocabulary(UNK)
            self.voc_ngrams = CategoricalVocabulary(UNK)
            self.voc_attributes = CategoricalVocabulary(UNK)

        if document:
            self.add_document(document)

    def dump(self, folder):
        with open(folder + '/voc_wordforms.p', 'wb') as f:
            pickle.dump(self.voc_wordforms, f)
        with open(folder + '/voc_ngrams.p', 'wb') as f:
            pickle.dump(self.voc_ngrams, f)
        with open(folder + '/voc_attributes.p', 'wb') as f:
            pickle.dump(self.voc_attributes, f)

    def add_document(self, document):
        self.voc_wordforms.freeze(False)
        for sentence in document.sentences:
            for token in sentence:
                wordform = token[WORDFORM_KEY]
                self.voc_wordforms.add(wordform)
                for i in range(1, MAX_NGRAMS + 1):
                    self.voc_ngrams.add(wordform[-i:])
                for key, value in token[ATTRIBUTE_KEY].items():
                    self.voc_attributes.add('{}:{}'.format(key, value))
        if REMOVE_RARE_WORDS:
            self.voc_wordforms.trim(2)  # Remove rare wordforms that are seen just once (less than 2) in training
        self.voc_wordforms.freeze()
        self.voc_ngrams.trim(2)  # Remove rare n-grams that are seen just once in training
        self.voc_ngrams.freeze()
        self.voc_attributes.freeze()
