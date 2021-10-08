#Crear un diccionario con todas las combinaciones de 3gram, donde la llave es el trigram
#y el valor es la frecuencia de repetición.
#X= Convertir luego el diccionario a un vector que contenga los valores de ese diccionario.
#y= Crear 
import numpy as np
# import itertools
from tweetlib.singleton import Utils 

# dict_ngram = {

# }
#Permutaciones con repeticiones https://stackoverflow.com/questions/3099987/generating-permutations-with-repetitions
# def dict_alpha_numeric(n):
#     list_alpha_numeric = 'abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTÁÉÍÓÚáéíóú1234567890@#`~ªº!"·$%&/()=?¿¡;,.:_-\'"]\[}{|<>'
#     result_dict = ["".join(p) for p in itertools.product(list_alpha_numeric, repeat=n)]

#     # print(result)
#     for i in result_dict:
#         dict_ngram[i] = 0
#     return dict_ngram

def char_grams(data_texts,n):
    vectors = []
    for text in data_texts:
        #Uno la lista de string sin espacios
        text_union = "".join(text)
        #Separo en ngram
        ngram = [text_union[i:i+n] for i in range(len(text_union)-n+1)]
        print(ngram)
        # dict_alpha_num = dict_alpha_numeric(n)
        dict_alpha_num = Utils.ngrams(n)
        for i in ngram:
            if i in dict_alpha_num:
                dict_alpha_num[i] += 1
                print(dict_alpha_num[i])
            else:
                print(i)
                print('El valor no existe en diccionario')
                continue
        vector_freq = freq_dict(dict_alpha_num)
        vectors.append(vector_freq)
    return vectors
# if __name__ == '__main__':
#     Char3Grams(b,n=3)

def freq_dict(dict_alpha_num):
    list_values = dict_alpha_num.values()
    vector = list(list_values)
    total_tokens = sum(vector)
    np_array = np.array(vector)
    vector_freq = np_array / total_tokens
    # print(vector_freq)
    return vector_freq


# if __name__ == '__main__':
#     dict_alpha()