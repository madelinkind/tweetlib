import itertools
from tweetlib.definitions import TaggingMethod, DictionarySize, Lang
from tweetlib.init_nlp import init_nlp

class Utils(object):
    ngram_dict = {}
    dict_ngram = {}
    dict_nlp = {}

    @staticmethod
    def ngrams(n):
        if n in Utils.ngram_dict:
            for key, _ in Utils.ngram_dict[n].items():
                Utils.ngram_dict[n][key] = 0
            return Utils.ngram_dict[n]
        else:
         
            list_alpha_numeric = 'abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚáéíóú1234567890@#? !;,.:_()|😥🥺😓😪😞😒😂🤣😅😊😆😁😄🙂😉😌😍🥰😘😗😚😋😛😝😜🤪🤓🤗📹😳❤️👏💪'

            result_dict = ["".join(p) for p in itertools.product(list_alpha_numeric, repeat=n)]

            for i in result_dict:
                Utils.dict_ngram[i] = 0

            # store ngrams in dict 'NGramUtils.ngram_dict'
            Utils.ngram_dict[n] = Utils.dict_ngram.copy()
            # return result
            return Utils.dict_ngram

    @staticmethod
    def load_nlp(nlp):
        if nlp.value in Utils.dict_nlp:
            return Utils.dict_nlp[nlp.value]

        if nlp.name == 'SPACY':
            Utils.dict_nlp[nlp.value] = init_nlp(TaggingMethod.SPACY, Lang.ES, size=DictionarySize.MEDIUM)
        elif nlp.name == 'STANZA':
            Utils.dict_nlp[nlp.value] = init_nlp(TaggingMethod.STANZA, Lang.ES, size=DictionarySize.MEDIUM)

        return Utils.dict_nlp[nlp.value]
