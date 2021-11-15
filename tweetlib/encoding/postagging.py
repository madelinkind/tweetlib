
from collections import Counter

import numpy as np
from stanza.models.common.doc import Document
# Using Stanza Batch
from typing import List
from stanza_batch import batch
from collections import Counter
import functools
from tweetlib.definitions import TaggingMethod, DictionarySize, Lang
from tweetlib.init_nlp import init_nlp
from tweetlib.singleton import Utils
from tweetlib.preprocessing.emoticons import count_emoticons, rm_emoticons
from tweetlib.preprocessing.hashtags import rm_hashtags, count_hashtags
from tweetlib.preprocessing.links import count_links, rm_links
from tweetlib.preprocessing.mentions import count_mentions, rm_mentions

# SPACY OR STANZA
#method: TaggingMethod
def postagging_encoding(data_texts, tagging_method, nlp):
    """[summary]

    Args:
        text (str): [description]
        method (TaggingMethod): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]

    """
    if tagging_method == 'SPACY':
        return spacy_postagging(data_texts, tagging_method, nlp)
    elif tagging_method == 'STANZA':
        return stanza_postagging(data_texts, tagging_method, nlp)
    else:
        raise Exception("You must enter a valid tagging method, See TaggingMethod.")

# SPACY_POSTAGGING
def spacy_postagging(data_texts: list, tagging_method, nlp):
    # Creating a spacy object
    vector_freq = freq_dict(data_texts, tagging_method, nlp)
    return vector_freq

# STANZA_POSTAGGING
def stanza_postagging(data_texts: str, tagging_method, nlp):
    vector_freq = freq_dict(data_texts, tagging_method, nlp)
    return vector_freq

# FREQUENCY DICTIONARY
def freq_dict(data_texts: list, tagging_method, nlp):
    vectors = []
    if tagging_method == 'SPACY':
        for idx, text in enumerate(data_texts):
            vector_new_tags = prep_text_new_tags(text)
            doc = nlp(" ".join(vector_new_tags[0]))

            freq_dict = Counter(([token.pos_ for token in doc]))
            vector = vector_freq(freq_dict, vector_new_tags)
            vectors.append(vector)
            # ---test---
            # for token in doc:
            #     print(token.text)
            #     print(token.pos_)
    elif tagging_method == 'STANZA':
        vectors = []
        list_stza = doc_withing_double_space(data_texts)
        #https://pypi.org/project/stanza-batch/
        for doc in batch(list_stza, nlp, batch_size=100):
            vector_new_tags = prep_text_new_tags(doc.text)
            if type(vector_new_tags[0]) == list:
                text = " ".join(vector_new_tags[0])
            sentence = join_sentences(text, nlp)
            if sentence == None:
                break
            freq_dict = Counter([word.upos for word in sentence])
            vector = vector_freq(freq_dict, vector_new_tags)
            vectors.append(vector)
    else: 
        raise Exception("You must enter a valid tagging method, See TaggingMethod.")
    return vectors

#Preprocesar texto para obtener la frecuencia de las nuevas etiquetas en dicho texto. 
# New tags (LINK, EMTC, MENT, HASH) 
def prep_text_new_tags(text):
    link = count_links(text)
    if link > 0:
        text = rm_links(text)
    emtc = count_emoticons(text)
    if emtc > 0:
        text = rm_emoticons(text)
    ment = count_mentions(text)
    if ment > 0:
        text = rm_mentions(text)
    htag = count_hashtags(text)
    if htag > 0:
        text = rm_hashtags(text)
    return text, link, emtc, ment, htag

#STANZA
def doc_withing_double_space(data_texts):
    lista_stza = []
    for text in data_texts:
        #Quitamos dobles espacios que pueda tener el texto
        if type(text) == list:
            text = " ".join(text)
        listado_stanza = ' '.join(text.split())
        lista_stza.append(listado_stanza)
    return lista_stza

def vector_freq(freq_dict, vector_new_tags):

    # freq_dict['EMAIL'] = vector_new_tags[1]
    freq_dict['LINK'] = vector_new_tags[1]
    freq_dict['EMTC'] = vector_new_tags[2]
    freq_dict['MENT'] = vector_new_tags[3]
    freq_dict['HASH'] = vector_new_tags[4]

    # vector = [freq_dict['ADJ'], freq_dict['ADP'], freq_dict['ADV'], freq_dict['AUX'], freq_dict['CONJ'], freq_dict['CCONJ'], freq_dict['DET'], freq_dict['INTJ'], freq_dict['NOUN'], freq_dict['NUM'], freq_dict['PART'], freq_dict['PRON'], freq_dict['PROPN'], freq_dict['PUNCT'], freq_dict['SCONJ'], freq_dict['SYM'], freq_dict['VERB'], freq_dict['X'], freq_dict['SPACE'], freq_dict['LINK'], freq_dict['EMTC'], freq_dict['EMAIL'], freq_dict['MENT'], freq_dict['HASH']]
    # vector = [freq_dict['ADJ'], freq_dict['ADP'], freq_dict['ADV'], freq_dict['AUX'], freq_dict['CCONJ'], freq_dict['DET'], freq_dict['INTJ'], freq_dict['NOUN'], freq_dict['NUM'], freq_dict['PRON'], freq_dict['PROPN'], freq_dict['PUNCT'], freq_dict['SCONJ'], freq_dict['SYM'], freq_dict['VERB'], freq_dict['LINK'], freq_dict['EMTC'], freq_dict['MENT'], freq_dict['HASH']]
    # vector = [freq_dict['ADJ'], freq_dict['ADP'], freq_dict['ADV'], freq_dict['AUX'], freq_dict['CONJ'], freq_dict['CCONJ'], freq_dict['DET'], freq_dict['INTJ'], freq_dict['NOUN'], freq_dict['NUM'], freq_dict['PART'], freq_dict['PRON'], freq_dict['PROPN'], freq_dict['PUNCT'], freq_dict['SCONJ'], freq_dict['SYM'], freq_dict['VERB'], freq_dict['X'], freq_dict['LINK'], freq_dict['EMTC'], freq_dict['MENT'], freq_dict['HASH']]
    vector = [freq_dict['ADJ'], freq_dict['ADP'], freq_dict['ADV'], freq_dict['AUX'], freq_dict['CCONJ'], freq_dict['DET'], freq_dict['INTJ'], freq_dict['NOUN'], freq_dict['NUM'], freq_dict['PART'], freq_dict['PRON'], freq_dict['PROPN'], freq_dict['PUNCT'], freq_dict['SCONJ'], freq_dict['SYM'], freq_dict['VERB'], freq_dict['SPACE'], freq_dict['LINK'], freq_dict['EMTC'], freq_dict['MENT'], freq_dict['HASH']]

    total_tokens = sum(freq_dict.values())
    np_array = np.array(vector)
    vector_freq = np_array/total_tokens
    return vector_freq

def join_sentences(text, nlp):
    total_words = []
    doc = nlp(text)
    
    for s in doc.sentences:
        total_words += s.words
    return total_words