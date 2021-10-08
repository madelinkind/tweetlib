from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod

nlp = Utils.load_nlp(TaggingMethod.SPACY)

# import re

# text = ["https://pp.com", "pp", "haciendo", "http://uno.dos", "fttp://test.tes", "prueba", "para", "eliminar", "links"]

# def RemoveLinks(text: list):
#     text_string = " ".join(text)
#     result = re.sub(r"http\S+", "", text_string)
#     result = result.strip()
#     list_text = list(result.split(" "))
#     print(list_text)

# if __name__ == '__main__':
#     RemoveLinks(text)

#-------------------------------------------------------------------------------------------------
#SECCIÓN DE PRUEBAS
#-------------------------------------------------------------------------------------------------

# import spacy
# import es_core_news_lg
# import stanza

#-------------------------------------------------------------------------------------------------
#SECCIÓN DATASET UTILIZADO
#-------------------------------------------------------------------------------------------------

# text = ["@Made", "#PP", "https://pp.com", "@Madelin", "#pepe", "Mucha", "mano", ",", "técnica", "en", "productos", "http://pp.com,"]
# text0 = ["ftp://pp.com", "Monitorización", "", "monitorizando", "realizando", "realización", "estudio", "emoticons", "😊", "😋", "🥰", "😚", ":)", "❥", "⚤", " ✱"]
# text1 = ["ftps://pp.com", "Ser", "e@e.o", "creativo", "significa", "estar", "enamorado", "de", "la", "vida"]
# text2 = ["No", "temas", "a", "la", "fttp://pp.com", "perfección", ":", "nunca", "ma@gmail.com", "la", "alcanzarás"]
# text3 = ["tt://pp.com", "Cualquier", "actividad", "se", "vuelve", "creativa"]
# text4 = ["Las", "manos", "realizan", "grandes", "labores", "ftp://pp.com"]

# def get_list_texts():
#     return [
#         text,
#         text0,
#         text1,
#         text2,
#         text3,
#         text4
#     ]

# list_texts = get_list_texts()

# Spacy
def rm_links (text: list):
    """Elimina los links de un corpus

    Args:
        text (list[str]): Lista de tokens.
        nlp ([type]): Biblioteca de spacy, para utilizar sus atributos, en este caso "like_url".

    Returns:
        list[str]: Lista sin links.
    """

    text_within_links = []
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
        
    text_within_links = [str(word) for word in doc if not word.like_url]
    return text_within_links

def count_links(text: list):
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    list_links = [word for word in doc if word.like_url]
    return len(list_links)

# if __name__ == '__main__':
#     RemoveLinks(list_texts)