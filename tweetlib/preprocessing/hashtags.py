import spacy
# import es_core_news_md

# text = "Casco para el US GP . \n   Helmet for the US GP . \n   # indy500tribute # # fans # thanks # mclaren …"

from spacy.matcher import Matcher
from spacy.tokens import Token

from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod

nlp = Utils.load_nlp(TaggingMethod.SPACY)

# Vector que devuelve una lista sin hashtag, otra con 
# el texto incluyendo los hashtag correctamente  y el total de hashtag del texto. 
#Cambio de nombre de rm_hashtags a tool_hashtags

def tool_hashtags(text: list):
    # a = 0
    # for t in text:
    #     if t == '#':
    #         pass
    # nlp = es_core_news_md.load()
    matcher = Matcher(nlp.vocab)
    matcher.add("HASHTAG", [[{"ORTH": "#"}, {"IS_ASCII": True}]])
    #Aqui me daba error, porque "is_hashtag" ya estaa agregado a Token, utilizo Token.has_extension 
    # para validar si ya se encuentra, de lo contrario se agrega. ver Documentación https://spacy.io/api/token 
    # if text == "Casco para el US GP . \n   Helmet for the US GP . \n   # indy500tribute # # fans # thanks # mclaren …":
        # a = 1
    if not Token.has_extension("is_hashtag"):
        Token.set_extension("is_hashtag", default=False)
    within_hash = []
    with_hash = [] 
    # doc = nlp(" ".join(text))
    doc = nlp(text)
    matches = matcher(doc)
    hashtags = []
    for match_id, start, end in matches:
        if doc.vocab.strings[match_id] == "HASHTAG":
            hashtags.append(doc[start:end])
    with doc.retokenize() as retokenizer:
        for span in hashtags:
            #Fue necesario insertar esta linea de codigo 
            # para este caso done toma un hashtags asi "# #" 
            # ya que que si el usuario se equivoca y pone sin darse cuenta esto,
            #  el codigo explota, porque el proximo hashtag lo toma con el caracter 
            # de # del hashtag anterior, y el error dice que no puede unir ese caracter 
            # pues ya ha sido utilizado.  
            if span.text == '# #':
                continue
            retokenizer.merge(span)
            for token in span:
                if not token._.is_hashtag:
                  token._.is_hashtag = True 
    for token in doc:
        # print(token.text, token._.is_hashtag)
        if not token._.is_hashtag:
            within_hash.append(token.text)
            with_hash.append(token.text)
        else:
            with_hash.append(token.text)
    vector = within_hash, with_hash, len(hashtags)

    return vector

def rm_hashtags(text):
    if type(text) == str:
        text_string = text
    else:
        text_string = " ".join(text)
    within_hashtag = tool_hashtags(text_string)
    return within_hashtag[0]

#Texto con hashtag, solucion al error que la libreria de spacy
# tiene a la hora de tokenizar, ya que divide el
# el caracter # de la palabra. Este proceso lo que
#  hace es devolver el texto con los hashtag correctamente representados
def fix_hashtags_in_text(text):
    with_hashtag = tool_hashtags(text)
    # print(with_hashtag[1])
    return with_hashtag[1]

def count_hashtags(text):
    if type(text) == str:
        text_string = text
    else:
        text_string = " ".join(text)
    count_hashtag_text = tool_hashtags(text_string)
    return count_hashtag_text[2] 


if __name__ == '__main__':
    pass
    # tool_hashtags(text)