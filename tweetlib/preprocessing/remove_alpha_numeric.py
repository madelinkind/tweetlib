from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod

nlp = Utils.load_nlp(TaggingMethod.SPACY)

def rm_alpha_numeric (text: list):
    """Eliminar todo lo que no sea caracteres alfabéticos  en el corpus

    Args:
        text (list): Lista de tokens.
        nlp ([type]): Biblioteca de spacy, para utilizar sus atributos, en este caso "is_alpha".

    Returns:
        [type]: Lista sin caracteres alfanuméricos.
    """
    text_within_characters = []
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    text_within_characters= [str(word) for word in doc if word.is_alpha]
    return text_within_characters

def count_alpha_numeric(text: list):
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    list_alpha_numeric = [word for word in doc if not word.is_alpha]
    # print(list_alpha_numeric)
    return len(list_alpha_numeric)