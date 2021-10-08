from tweetlib.encoding.postagging import postagging_encoding
from tweetlib.encoding.char2grams import char_2_gram
from tweetlib.encoding.char3grams import char_3_gram
from tweetlib.encoding.char4grams import char_4_gram
from tweetlib.encoding.all_chargrams import all_char_grams
from tweetlib.encoding.pos_all_chargram import pos_all_chargram
from tweetlib.encoding.word_embedding import preprocessing_encoding_for_bert
from tweetlib.definitions import EncodingMethod

dict_encoding = {
    EncodingMethod.BIGRAM: char_2_gram,
    EncodingMethod.TRIGRAM: char_3_gram,
    EncodingMethod.CUATRIGRAM: char_4_gram,
    EncodingMethod.POSTAGGING: postagging_encoding,
    EncodingMethod.ALL_CHARGRAM: all_char_grams,
    EncodingMethod.POS_ALL_CHARGRAM: pos_all_chargram,
    EncodingMethod.BERT: preprocessing_encoding_for_bert
}