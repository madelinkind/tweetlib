from enum import Enum

import stanza
import spacy
# , en-core-web-trf, es-core-news-md, es-core-news-sm, es-dep-news-trf
            # nlp = es-dep-news-trf.load()
from spacy import displacy
from .definitions import *

#INITIALIZE THE LIBRARY TO USE
def init_nlp(tagging_method: int, lang: int, size: int = DictionarySize.SMALL):
    if tagging_method == TaggingMethod.STANZA:
        if lang == Lang.ES:
#Le quito el lowercase (con este da problemas) y tokenize y lemma (se lo quito provicional). 
            nlp = stanza.Pipeline(lang='es', processors='tokenize,pos')
            return nlp
        elif lang == Lang.EN:
            nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
            return nlp
        else:
            raise Exception('')

    elif TaggingMethod.SPACY:
        if size == DictionarySize.SMALL and lang == Lang.ES:
            nlp = spacy.load("es_core_news_sm")
            nlp.add_pipe("emoji", first=True)
            return nlp
        elif size == DictionarySize.MEDIUM and lang == Lang.ES:
            nlp = spacy.load("es_core_news_md")
            nlp.add_pipe("emoji", first=True)
            return nlp
        elif size == DictionarySize.LARGE and lang == Lang.ES:
            nlp = spacy.load("es-dep-news-trf")
            nlp.add_pipe("emoji", first=True)
            return nlp
        elif size == DictionarySize.SMALL and lang == Lang.EN:
            nlp = spacy.load("en-core-web-trf")
            nlp.add_pipe("emoji", first=True)
            return nlp
        # elif size == DictionarySize.MEDIUM and lang == Lang.EN:
        #     nlp = en_core_web_md.load()
        #     return nlp
        # elif size == DictionarySize.LARGE and lang == Lang.EN:
        #     nlp = en_core_web_lg.load()
        #     return nlp
        else:
            raise ValueError("Could not build nlp with the given parameters")


# if not nlp.has_pipe("emoji"):
#     nlp.add_pipe("emoji", first=True)