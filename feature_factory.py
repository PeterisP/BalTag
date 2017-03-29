from collections import namedtuple, defaultdict
from lt_document import WORDFORM_KEY, WORDFORM_ORIGINAL_KEY, ATTRIBUTE_KEY
import numpy as np

FeatureVectors = namedtuple('FeatureVectors',
                            ['wordform_ids', 'wordform_embeddings', 'wordshape', 'ngrams', 'attribute_ids'])


class FeatureFactory(object):
    def __init__(self, vocabularies, embeddings):
        self._vocabularies = vocabularies
        if embeddings:
            # NB! hardcoded assumption that the <UNK> token is at ID 1, which is true for polyglot data only
            self._embeddings_lookup = defaultdict(lambda: 1, [(b, a) for a, b in enumerate(embeddings[0])])
            self._embeddings_table = embeddings[1]

    def wordform_vector_size(self):
        return len(self._vocabularies.voc_wordforms)  # one-hot vector describing the wordform

    def embedding_vector_size(self):
        return len(self._embeddings_table[0])

    def ngram_vector_size(self):
        return len(self._vocabularies.voc_ngrams)  # n-hot vector mapping to ngrams

    def attribute_vector_size(self):
        return len(self._vocabularies.voc_attributes)  # n-hot vector mapping to attribute-value pairs

    @staticmethod
    def wordshape_vector_size():
        return 1

    # convert a document to the appropriate numeric input and output vectors suitable for tagging
    def vectorize(self, document, configuration):
        return [self._vectorize_sentence(sentence, configuration)
                for sentence in document.sentences]

    def _vectorize_sentence(self, sentence, configuration):

        wordform_ids = np.array([self._vocabularies.voc_wordforms.get(token[WORDFORM_KEY]) for token in sentence])

        if configuration.input_embeddings:
            embedding_ids = [self._embeddings_lookup[token[WORDFORM_ORIGINAL_KEY]] for token in sentence]
            wordform_embeddings = self._embeddings_table[embedding_ids]

            # the tanh is neccessary if gensim embeddings are used, as they are in a range of -9 to +9
            # wordform_embeddings = np.tanh(self._embeddings[wordforms_filtered])  # Normalizing from [-x..+x] range to [0..1]
        else:
            wordform_embeddings = None

        if configuration.input_wordshape:
            wordshape = np.zeros([len(sentence), 1], dtype=np.float32)
            for tok_id, token in enumerate(sentence):
                if token[WORDFORM_ORIGINAL_KEY][0].isupper():  # TODO - placeholder, currently just check capitalization
                    wordshape[tok_id, 0] = 1
        else:
            wordshape = None

        if configuration.input_ngrams:
            ngrams = np.zeros([len(sentence), len(self._vocabularies.voc_ngrams)], dtype=np.float32)
            for tok_id, token in enumerate(sentence):
                wordform = token.get(WORDFORM_KEY)
                for i in range(1, configuration.max_ngrams + 1):
                    ngrams[tok_id, self._vocabularies.voc_ngrams.get(wordform[-i:])] = 1
        else:
            ngrams = None

        if True:  # output_attributes:
            attribute_ids = np.zeros([len(sentence), len(self._vocabularies.voc_attributes)], dtype=np.float32)
            for tok_id, token in enumerate(sentence):
                for key, value in token[ATTRIBUTE_KEY].items():
                    attribute_ids[tok_id, self._vocabularies.voc_attributes.get('{}:{}'.format(key, value))] = 1
        else:
            attribute_ids = None
        return FeatureVectors(wordform_ids, wordform_embeddings, wordshape, ngrams, attribute_ids)

    def dump(self, folder):
        self._vocabularies.dump(folder)
