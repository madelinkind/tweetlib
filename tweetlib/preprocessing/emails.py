from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod

nlp = Utils.load_nlp(TaggingMethod.SPACY)

# Spacy
def rm_emails (text: list):
    """Elimina las direcciones de correos de un corpus

    Args:
        text (list[str]): Lista de tokens
        nlp (spacy.lang.es.Spanish): Biblioteca de spacy, para utilizar sus atributos, en este caso "like_email"

    Returns:
        list[str]: Lista sin direcciones de correos.
    """

    text_within_emails = []

    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))

    text_within_emails = [str(word) for word in doc if not word.like_email]

    return text_within_emails

def count_email(text: list):
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    list_emails = [word for word in doc if word.like_email]
    return len(list_emails)