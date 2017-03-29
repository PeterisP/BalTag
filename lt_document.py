import json
import collections
import itertools

WORDFORM_KEY = 'wordform'
WORDFORM_ORIGINAL_KEY = 'wordform_original'
ATTRIBUTE_KEY = 'gold_attributes'


class AccuracyCounter():
    def __init__(self):
        self.c = collections.Counter()

    def add(self, gold, silver):
        self.c[gold == silver] += 1

    def add_b(self, boolean):
        self.c[boolean] += 1

    def average(self):
        total = sum(self.c.values())
        if not total:
            return 0
        return self.c[True] / total


# Wrapper of an annotated document
class LTDocument(object):
    # self.sentences structure : list of sentences; each sentence - list of tokens; each token - dict
    def __init__(self, filename, limit=None, pos_only=True):
        with open(filename, 'r') as f:
            self.sentences = [s for s in json.load(f) if s]
        if limit and limit < len(self.sentences):
            self.sentences = self.sentences[:limit]
        self.pos_only = pos_only
        self._preprocess()

    def _preprocess(self):
        for sentence in self.sentences:
            for token in sentence:
                if not token.get(WORDFORM_KEY):
                    print(json.dumps(token))
                    assert False
                self._simplify(token)

    def _simplify(self, token):
        token[WORDFORM_ORIGINAL_KEY] = token[WORDFORM_KEY]
        token[WORDFORM_KEY] = token[WORDFORM_KEY].lower()
        if token[ATTRIBUTE_KEY].get('unsorted_tag'):
            del token[ATTRIBUTE_KEY]['unsorted_tag']
        if self.pos_only:
            token[ATTRIBUTE_KEY] = {'POS': token[ATTRIBUTE_KEY].get('POS')}

    def output_tagged(self, silver_attributes, filename, vocabularies=None):
        attributes = AccuracyCounter()
        oov_attributes = AccuracyCounter()
        voc_attributes = AccuracyCounter()
        per_attribute = collections.defaultdict(AccuracyCounter)
        oov_per_attribute = collections.defaultdict(AccuracyCounter)
        voc_per_attribute = collections.defaultdict(AccuracyCounter)
        attribute_errors = collections.Counter()
        if not silver_attributes:
            silver_attributes = []

        with open(filename, 'w') as f:
            for sentence, sentence_attributes in itertools.zip_longest(self.sentences, silver_attributes):
                if not sentence_attributes:
                    sentence_attributes = []
                for token, silver_token_attributes in itertools.zip_longest(sentence, sentence_attributes):
                    gold_attrs = ','.join('{}:{}'.format(key, value) for key, value in token.get(ATTRIBUTE_KEY).items())
                    silver_attrs = ''
                    # Check the accuracy of predicted attributes
                    if silver_token_attributes:
                        errors = []
                        for key, gold_value in token.get(ATTRIBUTE_KEY).items():
                            silver_value = ''
                            best_confidence = 0
                            for silver_attribute, confidence in silver_token_attributes.items():
                                if silver_attribute.split(':')[0] != key:
                                    continue  # NB! we simply ignore the tagger's opinion on any attributes that are not relevant for this POS
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    silver_value = silver_attribute.split(':', maxsplit=1)[1]
                                    #                             print("{}: gold -'{}', silver -'{}' @ {}".format(key, gold_value, silver_value, best_confidence))
                            per_attribute[key].add(gold_value, silver_value)
                            if vocabularies and not vocabularies.voc_wordforms.get(token.get(WORDFORM_KEY)):
                                oov_per_attribute[key].add(gold_value, silver_value)
                            else:
                                voc_per_attribute[key].add(gold_value, silver_value)
                            if gold_value != silver_value:
                                errors.append('{}:{} instead of {}'.format(key, silver_value, gold_value))
                                attribute_errors['{}:{} instead of {}'.format(key, silver_value, gold_value)] += 1

                        attributes.add_b(not errors)
                        if vocabularies and not vocabularies.voc_wordforms.get(token.get(WORDFORM_KEY)):
                            oov_attributes.add_b(not errors)
                        else:
                            voc_attributes.add_b(not errors)
                        silver_attrs = '\t'.join(errors)

                    f.write('\t'.join(
                        [token.get(WORDFORM_ORIGINAL_KEY), gold_attrs, silver_attrs]) + '\n')
        print('Attribute accuracy:    {:.2%} ({:.2%} / {:.2%})'.format(attributes.average(), voc_attributes.average(), oov_attributes.average()))
        for key, counter in per_attribute.items():
            voc_counter = voc_per_attribute.get(key)
            if voc_counter:
                voc_counter_avg = voc_counter.average()
            else:
                voc_counter_avg = 0
            oov_counter = oov_per_attribute.get(key)
            if oov_counter:
                oov_counter_avg = oov_counter.average()
            else:
                oov_counter_avg = 0
            print('  {}: {:.2%} ({:.2%} / {:.2%})'.format(key, counter.average(), voc_counter_avg, oov_counter_avg))

        for key, count in attribute_errors.most_common(20):
            print('    {}: {}'.format(key, count))

