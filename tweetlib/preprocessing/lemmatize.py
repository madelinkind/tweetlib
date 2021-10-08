#Llevar la palapara a su estado base con SPACY (lemma)
# import es_core_news_lg
from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod
# text = ["Monitorización", "monitorizando", "realizando", "realización", "estudio", "emoticons", "😊", "😋", "🥰", "😚", ":)", "❥", "⚤", " ✱"]
nlp = Utils.load_nlp(TaggingMethod.SPACY)

def lemmatize(text: list):
    """Convierte cada token a su estado base con ayuda de la biblioteca SPACY

    Args:
        text (list[str]): Lista de tokens.
        nlp ([type]): Biblioteca de spacy, para utilizar sus atributos, en este caso "_lemma".

    Returns:
        list[str]: Lista lematizada.
    """
    lem_text = []
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    lem_text = [str(word.lemma_) for word in doc]
    # for word in tokens:
    #     lem_text.append(word.lemma_)
    return lem_text 

# if __name__ == '__main__':
#     Lemmatize(text)

#-------------------------------------------------------------------------------------------------
#SECCIÓN DE PRUEBAS
#-------------------------------------------------------------------------------------------------

# import spacy
# import es_core_news_lg
# import stanza

#-------------------------------------------------------------------------------------------------
#SECCIÓN DATASET UTILIZADO
#-------------------------------------------------------------------------------------------------

# text = ["Mucha", "mano", ",", "técnica", "en", "productos"]
# text0 = ["Monitorización", "monitorizando", "realizando", "realización", "estudio", "emoticons", "😊", "😋", "🥰", "😚", ":)", "❥", "⚤", " ✱"]
# text1 = ["Ser", "creativo", "significa", "estar", "enamorado", "de", "la", "vida"]
# text2 = ["No", "temas", "a", "la", "perfección", ":", "nunca", "la", "alcanzarás"]
# text3 = ["Cualquier", "actividad", "se", "vuelve", "creativa"]
# text4 = ["Las", "manos", "realizan", "grandes", "labores"]

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

#Spacy
# def Lemmatize(list_texts: list):
#     nlp = es_core_news_lg.load()
#     lem_text = []
#     for text in list_texts: 
#         doc = nlp(" ".join(text))
#         for word in doc:
#            print(f'word: {word} \tlemma: {word.lemma_}')

# Stanza
# def Lemmatize(list_texts: list):
#     nlp = stanza.Pipeline('es')
#     for text in list_texts: 
#         doc = nlp(" ".join(text))
#         for sent in doc.sentences:
#             for word in sent.words:
#                 print(f'word: {word.text} \tlemma: {word.lemma}') 

# if __name__ == '__main__':
#     Lemmatize(list_texts)

#-------------------------------------------------------------------------------------------------
#SECCIÓN RESULTADOS
#-------------------------------------------------------------------------------------------------

# Tras realizar pequeñas pruebas de comparación del lematizador de stanza y spacy,
# hemos llegado a la conclusión que para el idioma español, spacyy arroja mejores
# resultados. Por lo que se ha decidido trabajar con el lematizador de spacy, 
# para el preprocesamiento.
#Estas pruebas al ser escasas no son concluyentes.

#A continuación muestro algunos resultados

#Textos utilizados para realizar la comparativa

#TESTS
#STANZA MEJOR
# word: Mucha     lemma: mucho
# word: mano      lemma: mano
# word: ,         lemma: ,
# word: técnica   lemma: técnica
# word: en        lemma: en
# word: productos lemma: producto

#SPACY
# word: Mucha     lemma: Mucha
# word: mano      lemma: manir     MAL
# word: ,         lemma: ,
# word: técnica   lemma: técnico
# word: en        lemma: en
# word: productos lemma: producto
#---------------------------------

#STANZA
# word: Ser       lemma: ser
# word: creativo  lemma: creativo
# word: significa lemma: significar
# word: estar     lemma: estar
# word: enamorado lemma: enamorado
# word: de        lemma: de
# word: la        lemma: él
# word: vida      lemma: vida

#SPACY  MEJOR
# word: Ser       lemma: Ser
# word: creativo  lemma: creativo
# word: significa lemma: significar
# word: estar     lemma: estar
# word: enamorado lemma: enamorar
# word: de        lemma: de
# word: la        lemma: lo
# word: vida      lemma: vida
#---------------------------------

#STANZA
# word: No         lemma: no
# word: temas      lemma: tema
# word: a          lemma: a
# word: la         lemma: él
# word: perfección lemma: perfección
# word: :          lemma: :
# word: nunca      lemma: nunca
# word: la         lemma: él
# word: alcanzarás lemma: alcanzar

#SPACY  MEJOR
# word: No         lemma: No
# word: temas      lemma: temer
# word: a          lemma: a
# word: la         lemma: lo
# word: perfección lemma: perfección
# word: :          lemma: :
# word: nunca      lemma: nunca
# word: la         lemma: lo
# word: alcanzarás lemma: alcanzar
#---------------------------------

#STANZA
# word: Cualquier  lemma: cualquiera
# word: actividad  lemma: actividad
# word: se         lemma: él
# word: vuelve     lemma: volver
# word: creativa   lemma: creativo

#SPACY   MEJOR
# word: Cualquier    lemma: Cualquier
# word: actividad    lemma: actividad
# word: se           lemma: se
# word: vuelve       lemma: volver
# word: creativa     lemma: creativo
#---------------------------------
#STANZA
# word: Las       lemma: él
# word: manos     lemma: mano
# word: realizan  lemma: realizar
# word: grandes   lemma: grande
# word: labores   lemma: labor

#SPACY MEJOR
# word: Las       lemma: Las
# word: manos     lemma: mano
# word: realizan  lemma: realizar
# word: grandes   lemma: grande
# word: labores   lemma: laborar