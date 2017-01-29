from pathlib import Path
from collections import Counter, defaultdict
import re
import json

# Uses MATAS v0.2 Lithuanian corpus; not distributed here, should be available in CLARIN
CORPUS_FOLDER_LT = '/Users/pet/Documents/LT korpuss/'

c = Counter()


def convert_attributes(tags):
    attributes = defaultdict(str)
    # Based on http://donelaitis.vdu.lt/main.php?id=4&nr=7_1
    for tag in tags.split():
        if tag == 'jngt':
            attributes['POS'] = 'Conjunction'
        elif tag == 'idJngt':
            attributes['POS'] = 'ConjunctiveIdiom'
        elif tag == 'idPrln':
            attributes['POS'] = 'PrepositionalIdiom'
        elif tag == 'prln':
            attributes['POS'] = 'Preposition'
        elif tag == 'sntrmp':
            attributes['POS'] = 'Abbreviation'
        elif tag == 'akronim':
            attributes['POS'] = 'Acronym'
        elif tag == 'prvks':
            attributes['POS'] = 'Adverb'
        elif tag == 'dll':
            attributes['POS'] = 'Particle'
        elif tag == 'jstk':
            attributes['POS'] = 'Interjection'
        elif tag == 'ištk':
            attributes['POS'] = 'Onomatopoetic'
        elif tag == 'teig':
            attributes['Negation'] = 'Positive'
        elif tag == 'neig':
            attributes['Negation'] = 'Negative'
        elif tag in ['nelygin.l', 'aukšč.l', 'aukštėlesn.l', 'aukštesn.l']:
            attributes['Degree'] = tag
        elif tag == 'dktv':
            attributes['POS'] = 'Noun'
        elif tag == 'bdvr':
            attributes['POS'] = 'Adjective'
        elif tag == 'tikr.dktv':
            attributes['POS'] = 'ProperNoun'
        elif tag == 'tikr.dktv2':
            attributes['POS'] = 'ProperNoun2'
        elif tag == 'sktv':
            attributes['POS'] = 'Number'
        elif tag == 'rom.skaič':
            attributes['POS'] = 'RomanNumber'
        elif tag == 'vyr.gim':
            attributes['Gender'] = 'Masculine'
        elif tag == 'mot.gim':
            attributes['Gender'] = 'Feminine'
        elif tag == 'bevrd.gim':
            attributes['Gender'] = 'Bevardė'
        elif tag == 'bendr.gim':
            attributes['Gender'] = 'Bendroje'
        elif tag == 'vnsk':
            attributes['Number'] = 'Singular'
        elif tag == 'dgsk':
            attributes['Number'] = 'Plural'
        elif tag == 'dvisk':
            attributes['Number'] = 'Dual'
        elif tag in ['K', 'V', 'Vt', 'Įn', 'G', 'N', 'Š', 'Il']:
            attributes['Case'] = tag
        elif tag == 'bndr':
            attributes['POS'] = 'Infinitive'
        elif tag == 'būdn':
            attributes['POS'] = 'Infinitive2'
        elif tag == 'vksm':
            attributes['POS'] = 'Verb'
        elif tag in ['įvardž', 'neįvardž']:
            attributes['Definiteness'] = tag
        elif tag == 'įvrd':
            attributes['POS'] = 'Pronoun'
        elif tag == 'dlv':
            attributes['POS'] = 'Participle'
        elif tag == 'padlv':
            attributes['POS'] = 'Gerund'
        elif tag == 'psdlv':
            attributes['POS'] = 'Halfparticiple'
        elif tag in ['Iasm', 'IIasm', 'IIIasm']:
            attributes['asm'] = tag
        elif tag in ['sngr', 'nesngr']:
            attributes['sngr'] = tag
        elif tag == 'nesangr':
            attributes['sngr'] = 'nesngr'
        elif tag in ['veik.r', 'neveik.r']:
            attributes['veik.r'] = tag
        elif tag in ['tiesiog.nuos', 'tariog.nuos', 'tariam.nuos', 'liep.nuos']:
            attributes['nuos'] = tag
        elif tag in ['esam.l', 'būs.l', 'būt.l', 'būt.kart.l', 'būt.d.l', 'būt.k.l']:
            attributes['Tense'] = tag
        elif tag in ['daugin', 'kelintin', 'kiekin', 'kuopin']:
            attributes['NumberType'] = tag
        elif tag == 'reikiamyb':
            attributes['Debitive'] = 'Yes'
        else:
            attributes['unsorted_tag'] += ' ' + tag
    for (key, value) in attributes.items():
        c[key + ':' + value] += 1
    return attributes


def load_lt_file(file):
    print(file)
    sentences = []
    current_sentence = []
    with file.open() as corpus_file:
        for line in corpus_file:
            if line.startswith(u'\ufeff'):
                line = line[1:]
            # NB - horrible hacks here to parse this XML-like format that's far from valid XML

            if line.startswith('<word'):
                match = re.match('^<word="(.*)" lem?ma="(.*?)"? type="?(.*?)"?>?,?\n?$', line, re.UNICODE)
                if match:
                    wordform = match.group(1)
                    lemma = match.group(2)
                    tags = match.group(3)
                    # print('{}\t{}\t{}'.format(word, lemma, tags))
                    word = {
                        'wordform': wordform,
                        # 'gold_lemma' : lemma,
                        'gold_attributes': convert_attributes(tags),
                        'tags': tags
                    }
                    current_sentence.append(word)
                else:
                    pass
                    # print('Slikta rinda |{}|'.format(line))
            elif line.startswith('<space>') or line == '<tab>\n':
                pass
                # current_sentence += ' '
            elif line.startswith('<sep'):
                match = re.match('^<sep="(.*)">$', line, re.UNICODE)
                if match:
                    sep = match.group(1)
                    word = {
                        'wordform': sep,
                        'gold_attributes': {'POS': 'Sep'}
                    }
                    current_sentence.append(word)
                else:
                    pass
                    # print('Slikta rinda |{}|'.format(line))
            elif line.startswith('<doc'):
                continue
            elif line.startswith('<number>') or line.startswith('</number>') or \
                    line.startswith('<docDate>') or line.startswith('</docDate>') or \
                    line.startswith('<doc>') or line.startswith('</doc>') or \
                    line.startswith('<author>') or line.startswith('</author>') or \
                    line.startswith('<date>') or line.startswith('</date>') or line.startswith('</docGroup>'):
                # In this case the actual tokens are inside this tag and we can ignore the wrapper
                continue
            elif line.startswith('<head>') or line.startswith('</head>') or line.startswith('<heading>') or \
                    line.startswith('<div>') or line.startswith('</div>') or line.startswith('<title>') or \
                    line.startswith('</title>') or line.startswith('<body>') or line.startswith('</heading>'):
                # In this case the actual tokens are inside this tag and we can ignore the number wrapper
                continue
            elif line.startswith('<number'):
                match = re.match('^<number="(.*)">$', line, re.UNICODE)
                if match:
                    number = match.group(1)
                    word = {
                        'wordform': number,
                        'gold_attributes': {'POS': 'Number'}
                    }
                    current_sentence.append(word)
                else:
                    print('Slikta rinda |{}|'.format(line))
            elif line.startswith('<fragment'):
                    match = re.match('^<fragment>(.*)</fragment>$', line, re.UNICODE)
                    if match:
                        fragment = match.group(1)
                        word = {
                            'wordform': fragment,
                            'gold_attributes': {'POS': 'Fragment'}
                        }
                        current_sentence.append(word)
                    else:
                        print('Slikta rinda |{}|'.format(line))
            elif line.startswith('<letter'):
                    match = re.match('^<letter="(.*)">$', line, re.UNICODE)
                    if match:
                        letter = match.group(1)
                        word = {
                            'wordform': letter,
                            'gold_attributes': {'POS': 'Letter'}
                        }
                        current_sentence.append(word)
                    else:
                        print('Slikta rinda |{}|'.format(line))
            elif line.startswith('<syllable'):
                    match = re.match('^<syllable="(.*)">$', line, re.UNICODE)
                    if match:
                        syllable = match.group(1)
                        word = {
                            'wordform': syllable,
                            'gold_attributes': {'POS': 'Syllable'}
                        }
                        current_sentence.append(word)
                    else:
                        print('Slikta rinda |{}|'.format(line))
            elif line.startswith('<foreign'):
                match = re.match('^<foreign lang?="(.*)">(.*)</forei(gn|ng)>$', line, re.UNICODE)
                if match:
                    lang = match.group(1)
                    foreign = match.group(2)
                    word = {
                        'wordform': foreign,
                        'gold_attributes': {'POS': 'Foreign', 'Lang': lang}
                    }
                    current_sentence.append(word)
                else:
                    pass
                    # print('Slikta rinda |{}|'.format(line))
            elif line.startswith('<p>'):
                sentences.append(current_sentence)
                current_sentence = []
            else:
                # print(line)
                pass

    # for nr, sentence in enumerate(sentences):
    #     print(sentence)
    #     if nr > 3:
    #         break

    return sentences


def load_lt_data(path, result_filename):
    sentences = []
    for path in Path(path).iterdir():
        if 'CLARIN-LT_Terms-of-Service.txt' in str(path):
            continue
        if path.suffix == '.txt':
            sentences.extend(load_lt_file(path))

    # for key, count in c.most_common():
    #     print('{} : {}'.format(key, count))
    with open(result_filename, 'w', encoding='utf8') as outfile:
        json.dump(sentences, outfile, indent=4, ensure_ascii=False)


# ------ main ----
load_lt_data(CORPUS_FOLDER_LT + 'dev', 'data/dev.json')
load_lt_data(CORPUS_FOLDER_LT + 'test', 'data/test.json')
load_lt_data(CORPUS_FOLDER_LT + 'train', 'data/train.json')
print('Done!')