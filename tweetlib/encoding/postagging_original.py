
from collections import Counter

import numpy as np

from tweetlib.definitions import TaggingMethod, DictionarySize, Lang
from tweetlib.init_nlp import init_nlp
from tweetlib.test_data import SpanishTextSamples

#INITIALIZE THE LIBRARY TO USE
# nlp = init_nlp(TaggingMethod.STANZA, Lang.ES, size=DictionarySize.MEDIUM)
nlp = init_nlp(TaggingMethod.SPACY, Lang.ES, size=DictionarySize.MEDIUM)

# SPACY OR STANZA
#method: TaggingMethod
def postagging_encoding(text: list, method: TaggingMethod, nlp):
    """[summary]

    Args:
        text (str): [description]
        method (TaggingMethod): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]

    """
    if method == TaggingMethod.SPACY:
        return spacy_postagging(text, TaggingMethod.SPACY)
    elif method == TaggingMethod.STANZA:
        return stanza_postagging(text, TaggingMethod.STANZA)
    else:
        raise Exception("You must enter a valid tagging method, See TaggingMethod.")

# SPACY_POSTAGGING
def spacy_postagging(text: list, method: TaggingMethod):
    # Creating a spacy object
    vector_freq = freq_dict(text, method)
    return vector_freq

# STANZA_POSTAGGING
def stanza_postagging(text: str, method: TaggingMethod):
    vector_freq = freq_dict(text, method)
    return vector_freq

# FREQUENCY DICTIONARY
def freq_dict(text: list, method: TaggingMethod):
    # doc = nlp(" ".join(text))
    doc = nlp(text)

    if method == TaggingMethod.SPACY:
        freq_dict = Counter(([token.pos_ for token in doc]))
        #---test---
        for token in doc:
            print(token.text)
            print(token.pos_)
    elif method == TaggingMethod.STANZA:
        for sent in doc.sentences:
            freq_dict = Counter([word.upos for word in sent.words])
            #---test---
            # for word in sent.words:
            #     print(f'word: {word.text} \tlemma: {word.lemma} \tpos: {word.upos}')
    else:
        raise Exception("You must enter a valid tagging method, See TaggingMethod.")

    vector = [freq_dict['ADJ'], freq_dict['ADP'], freq_dict['ADV'], freq_dict['AUX'], freq_dict['CONJ'], freq_dict['CCONJ'], freq_dict['DET'], freq_dict['INTJ'], freq_dict['NOUN'], freq_dict['NUM'], freq_dict['PART'], freq_dict['PRON'], freq_dict['PROPN'], freq_dict['PUNCT'], freq_dict['SCONJ'], freq_dict['SYM'], freq_dict['VERB'], freq_dict['X'], freq_dict['SPACE']]
    total_tokens = sum(freq_dict.values())
    np_array = np.array(vector)
    vector_freq = np_array/total_tokens
    return vector_freq