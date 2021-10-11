#Crear un diccionario con todas las combinaciones de 3gram, donde la llave es el trigram
#y el valor es la frecuencia de repetición.
#X= Convertir luego el diccionario a un vector que contenga los valores de ese diccionario.
#y= Crear 
import numpy as np
# import itertools
from tweetlib.singleton import Utils 

def char_grams(data_texts,n):
    vectors = []
    for text in data_texts:
        #Uno la lista de string sin espacios
        text_union = "".join(text)
        #Separo en ngram
        ngram = [text_union[i:i+n] for i in range(len(text_union)-n+1)]
        # print(ngram)
        # dict_alpha_num = dict_alpha_numeric(n)
        dict_alpha_num = Utils.ngrams(n)
        for i in ngram:
            if i in dict_alpha_num:
                dict_alpha_num[i] += 1
                # print(dict_alpha_num[i])
            else:
                continue
        vector_freq = freq_dict(dict_alpha_num)
        vectors.append(vector_freq)
    return vectors

def freq_dict(dict_alpha_num):
    list_values = dict_alpha_num.values()
    vector = list(list_values)
    total_tokens = sum(vector)
    np_array = np.array(vector)
    vector_freq = np_array / total_tokens
    return vector_freq
