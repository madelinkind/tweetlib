# Spacy
from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod

nlp = Utils.load_nlp(TaggingMethod.SPACY)

def rm_num (text: list):
    """Elimina los números de un corpus

    Args:
        text (list[str]): Lista de tokens
        nlp (spacy.lang.es.Spanish): Biblioteca de spacy, para utilizar sus atributos, en este caso "like_num"

    Returns:
        list[str]: Lista sin números.
    """
    text_within_num = []
    doc = nlp(" ".join(text))
    # doc = nlp(text)
    text_within_num = [str(word) for word in doc if not word.like_num]
    return text_within_num

def count_nums(text: list):
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    list_nums = [word for word in doc if word.like_num]
    return len(list_nums)