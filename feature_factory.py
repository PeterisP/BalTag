from collections import namedtuple
from vocabularies import UNK, MAX_NGRAMS
from lt_document import WORDFORM_KEY, WORDFORM_ORIGINAL_KEY, ATTRIBUTE_KEY
import numpy as np

FeatureVectors = namedtuple('FeatureVectors',
                            ['wordform_ids', 'wordform_embeddings', 'wordshape', 'ngrams', 'attribute_ids'])


class FeatureFactory(object):
    def __init__(self, vocabularies, embeddings):
        self._vocabularies = vocabularies
        self._embeddings = embeddings
        # As gensim models crash when requesting OOV words, we add an UNK token as id 0
        if self._embeddings and UNK not in self._embeddings:
            gensim_add_zero_word(self._embeddings, UNK)

    def wordform_vector_size(self):
        return len(self._vocabularies.voc_wordforms)  # one-hot vector describing the wordform

    def embedding_vector_size(self):
        return self._embeddings.vector_size

    def ngram_vector_size(self):
        return len(self._vocabularies.voc_ngrams)  # n-hot vector mapping to ngrams

    def attribute_vector_size(self):
        return len(self._vocabularies.voc_attributes)  # n-hot vector mapping to attribute-value pairs

    @staticmethod
    def wordshape_vector_size():
        return 1

    # convert a document to the appropriate numeric input and output vectors suitable for tagging
    def vectorize(self, document, use_wordform_embeddings, use_wordshape, use_ngrams, output_attributes):
        return [self._vectorize_sentence(sentence, use_wordform_embeddings, use_wordshape, use_ngrams, output_attributes)
                for sentence in document.sentences]

    def _vectorize_sentence(self, sentence, use_wordform_embeddings, use_wordshape, use_ngrams, output_attributes):

        wordform_ids = np.array([self._vocabularies.voc_wordforms.get(token[WORDFORM_KEY]) for token in sentence])

        if use_wordform_embeddings:
            wordforms_filtered = [token[WORDFORM_KEY] if token[WORDFORM_KEY] in self._embeddings else UNK for token in
                                  sentence]
            wordform_embeddings = np.tanh(
                self._embeddings[wordforms_filtered])  # Normalizing from [-x..+x] range to [0..1]
        else:
            wordform_embeddings = None

        if use_wordshape:
            wordshape = np.zeros([len(sentence), 1], dtype=np.float32)
            for tok_id, token in enumerate(sentence):
                if token[WORDFORM_ORIGINAL_KEY][0].isupper():  # TODO - placeholder, currently just check capitalization
                    wordshape[tok_id, 0] = 1
        else:
            wordshape = None

        if use_ngrams:
            ngrams = np.zeros([len(sentence), len(self._vocabularies.voc_ngrams)], dtype=np.float32)
            for tok_id, token in enumerate(sentence):
                wordform = token.get(WORDFORM_KEY)
                for i in range(1, MAX_NGRAMS + 1):
                    ngrams[tok_id, self._vocabularies.voc_ngrams.get(wordform[-i:])] = 1
        else:
            ngrams = None

        if output_attributes:
            attribute_ids = np.zeros([len(sentence), len(self._vocabularies.voc_attributes)], dtype=np.float32)
            for tok_id, token in enumerate(sentence):
                for key, value in token[ATTRIBUTE_KEY].items():
                    attribute_ids[tok_id, self._vocabularies.voc_attributes.get('{}:{}'.format(key, value))] = 1
        else:
            attribute_ids = None
        return FeatureVectors(wordform_ids, wordform_embeddings, wordshape, ngrams, attribute_ids)

    def dump(self, folder):
        self._vocabularies.dump(folder)


# Helper code to allow adding a new entry to a gensim vocabulary after generating it
# Vocab class stub / constructor taken from gensim code
class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).
    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


def gensim_add_zero_word(model, word):
    word_id = len(model.vocab)
    model.vocab[word] = Vocab(index=word_id, count=1)
    model.syn0 = np.append(model.syn0, [np.zeros(model.vector_size, dtype=np.float32)], 0)
    model.index2word.append(word)