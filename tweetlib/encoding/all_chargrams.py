from tweetlib.encoding.char2grams import char_2_gram
from tweetlib.encoding.char3grams import char_3_gram
import numpy as np
# from tweetlib.encoding.char4grams import char_4_gram

def all_char_grams(data_texts):
    bi = char_2_gram(data_texts)
    tri = char_3_gram(data_texts)
    tup = bi, tri
    vector_concat = np.hstack(tup)
    # cuatri = char_4_gram(data_texts)
    # for idx, bt in enumerate(data_texts):
        # vector_concat[idx] = np.concatenate(bi[idx], tri[idx])

    return vector_concat