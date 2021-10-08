
from collections import Counter

import numpy as np

from tweetlib.definitions import TaggingMethod, DictionarySize, Lang
from tweetlib.init_nlp import init_nlp
from tweetlib.test_data import SpanishTextSamples
from tweetlib.preprocessing.emails import count_email, rm_emails
from tweetlib.preprocessing.emoticons import count_emoticons, rm_emoticons
from tweetlib.preprocessing.hashtags import rm_hashtags, count_hashtags
from tweetlib.preprocessing.links import count_links, rm_links
from tweetlib.preprocessing.mentions import count_mentions, rm_mentions
from tweetlib.preprocessing.num import count_nums, rm_num
from tweetlib.preprocessing.punct import count_puncts, rm_punct

#INITIALIZE THE LIBRARY TO USE
# nlp = init_nlp(TaggingMethod.STANZA, Lang.ES, size=DictionarySize.MEDIUM)
nlp = init_nlp(TaggingMethod.SPACY, Lang.ES, size=DictionarySize.MEDIUM)

# SPACY OR STANZA
#method: TaggingMethod
def postagging_encoding(text: str, method: TaggingMethod, nlp, listado_stanza):
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
        return spacy_postagging(text, TaggingMethod.SPACY, nlp)
    elif method == TaggingMethod.STANZA:
        return stanza_postagging(text, TaggingMethod.STANZA, nlp, listado_stanza)
    else:
        raise Exception("You must enter a valid tagging method, See TaggingMethod.")

# SPACY_POSTAGGING
def spacy_postagging(text: list, method: TaggingMethod, nlp):
    # Creating a spacy object
    vector_freq = freq_dict(text, method, nlp)
    return vector_freq

# STANZA_POSTAGGING
def stanza_postagging(text: str, method: TaggingMethod, nlp, listado_stanza):
    vector_freq = freq_dict(text, method, nlp, listado_stanza)
    return vector_freq

# FREQUENCY DICTIONARY
def freq_dict(text: list, method: TaggingMethod, nlp, listado_stanza):
    # doc = nlp(" ".join(text))
    # freq_dict['NUM'] = count_nums(doc)
    # freq_dict['PUNCT'] = count_puncts(doc)
    text_filter = text
    # htag = count_hashtags(text_filter, nlp)
    # if htag > 0:
    #     text_filter = rm_hashtags(text_filter, nlp)

    #Luego de probar stanza descomentariar
    email = count_email(text_filter, nlp)
    if email > 0:
        text_filter = rm_emails(text_filter, nlp)
    link = count_links(text_filter, nlp)
    if link > 0:
        text_filter = rm_links(text_filter, nlp)
    emtc = count_emoticons(text_filter, nlp)
    if emtc > 0:
        text_filter = rm_emoticons(text_filter, nlp)
    ment = count_mentions(text_filter)
    if ment > 0:
        text_filter = rm_mentions(text_filter)
    htag = count_hashtags(text_filter, nlp)
    if htag > 0:
        text_filter = rm_hashtags(text_filter, nlp)

    doc = nlp(" ".join(text_filter))
    # doc = nlp(listado_stanza)

    if method == TaggingMethod.SPACY:
        freq_dict = Counter(([token.pos_ for token in doc]))
        # ---test---
        # for token in doc:
        #     print(token.text)
        #     print(token.pos_)
    elif method == TaggingMethod.STANZA:
        vectors = []
        for sent in doc.sentences:
            freq_dict = Counter([word.upos for word in sent.words])
            vector = [freq_dict['ADJ'], freq_dict['ADP'], freq_dict['ADV'], freq_dict['AUX'], freq_dict['CONJ'], freq_dict['CCONJ'], freq_dict['DET'], freq_dict['INTJ'], freq_dict['NOUN'], freq_dict['NUM'], freq_dict['PART'], freq_dict['PRON'], freq_dict['PROPN'], freq_dict['PUNCT'], freq_dict['SCONJ'], freq_dict['SYM'], freq_dict['VERB'], freq_dict['X'], freq_dict['SPACE'], freq_dict['LINK'], freq_dict['EMTC'], freq_dict['EMAIL'], freq_dict['MENT'], freq_dict['HASH']]
            total_tokens = sum(freq_dict.values())
            np_array = np.array(vector)
            vector_freq = np_array/total_tokens
            #Add new
            vectors.append(vector_freq)
            #---test---
            # for word in sent.words:
            #     print(f'word: {word.text} \tlemma: {word.lemma} \tpos: {word.upos}')
    else:
        raise Exception("You must enter a valid tagging method, See TaggingMethod.")
    #Luego de probar stanza descomentariar
    freq_dict['EMAIL'] = email
    freq_dict['LINK'] = link
    freq_dict['EMTC'] = emtc
    freq_dict['MENT'] = ment
    freq_dict['HASH'] = htag

    #Decomentariar una vez termine con STANZA
    vector = [freq_dict['ADJ'], freq_dict['ADP'], freq_dict['ADV'], freq_dict['AUX'], freq_dict['CONJ'], freq_dict['CCONJ'], freq_dict['DET'], freq_dict['INTJ'], freq_dict['NOUN'], freq_dict['NUM'], freq_dict['PART'], freq_dict['PRON'], freq_dict['PROPN'], freq_dict['PUNCT'], freq_dict['SCONJ'], freq_dict['SYM'], freq_dict['VERB'], freq_dict['X'], freq_dict['SPACE'], freq_dict['LINK'], freq_dict['EMTC'], freq_dict['EMAIL'], freq_dict['MENT'], freq_dict['HASH']]
    total_tokens = sum(freq_dict.values())
    np_array = np.array(vector)
    vector_freq = np_array/total_tokens
    return vector_freq
    return vectors

#Preprocesar texto para obtener la frecuencia de las nuevas etiquetas en dicho texto. 
# New tags (EMAIL, LINK, EMTC, MENT, HASH) 
def prep_text_new_tags(text, nlp_spacy):
    email = count_email(text, nlp_spacy)
    if email > 0:
        text = rm_emails(text, nlp_spacy)
    link = count_links(text, nlp_spacy)
    if link > 0:
        text = rm_links(text, nlp_spacy)
    emtc = count_emoticons(text, nlp_spacy)
    if emtc > 0:
        text = rm_emoticons(text, nlp_spacy)
    ment = count_mentions(text)
    if ment > 0:
        text = rm_mentions(text)
    htag = count_hashtags(text, nlp_spacy)
    if htag > 0:
        text = rm_hashtags(text, nlp_spacy)
    return text, email, link, emtc, ment, htag
