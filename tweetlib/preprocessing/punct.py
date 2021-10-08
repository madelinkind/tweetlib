# Spacy
from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod

nlp = Utils.load_nlp(TaggingMethod.SPACY)

def rm_punct(text: list):
    """Elimina los signos de puntuación de un corpus

    Args:
        text (list[str]): Lista de tokens
        nlp (spacy.lang.es.Spanish): Biblioteca de spacy, para utilizar sus atributos, en este caso "is_punct"

    Returns:
        list[str]: Lista sin signos de puntuación.
    """
    text_within_punct = []
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    text_within_punct = [str(word) for word in doc if not word.is_punct]
    return text_within_punct

def count_puncts(text: list):
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    list_puncts = [word for word in doc if word.is_punct]
    return len(list_puncts)