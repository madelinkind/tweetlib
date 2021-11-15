from spacy.lang.es.stop_words import STOP_WORDS

# para observar las palabras dentro de la lista lo único que se #   # tiene que hacer eso imprimir
#print(list(STOP_WORDS)[:30])

# text = ['además', 'ir', ':(', ':)', '']

def rm_stop_words(text: list):
    list_text = []
    if type(text) == str:
        text = text.split(" ")
    #Para filtrar stopwords
    list_text = [str(word) for word in text if not word in STOP_WORDS]
    # for word in text:
    #     if not word in STOP_WORDS:
    #         list_text.append(word)
    return list_text

# if __name__ == '__main__':
#     Stop_words(text)