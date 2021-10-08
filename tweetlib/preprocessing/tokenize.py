from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod
# from tweetlib.definitions import TaggingMethod, DictionarySize, Lang
# from tweetlib.init_nlp import init_nlp
nlp = Utils.load_nlp(TaggingMethod.SPACY)
#Tokenizar el texto
def tokenize(text: str):
    """Convierte un string en una lista de tokens

    Args:
        text (str): Cadena de texto.
        nlp ([type]): Biblioteca de spacy, para utilizar sus atributos, en este caso "like_email".

    Returns:
        list[str]: Lista tokenizada.
    """
    # Create list of word tokens
    token_list = []
    # nlp = init_nlp(TaggingMethod.SPACY, Lang.ES, size=DictionarySize.MEDIUM)
    doc = nlp(text)
    token_list = [token.text for token in doc]
    # for token in doc:
    #     token_list.append(token.text)
    return token_list