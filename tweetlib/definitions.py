from enum import Enum

# TYPES OF LIBRARIES
class TaggingMethod(Enum):
    STANZA = 1
    SPACY = 2

# VOCABULARY SIZE
class DictionarySize(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

# TYPE OF LANGUAGE
class Lang(Enum):
    ES = 1
    EN = 2

# TYPE OF USER
class TypeDataSet():
    politico = 'politico'
    artista = 'artista'
    deportista = 'deportista'
    youtuber = 'youtuber'
    all_results = 'todos'

# TYPE OF DATASET
# class DATASET(Enum):
#     politico = 1
#     artista = 2
#     deportista = 3
#     youtuber = 4
#     all_results = 5

# TYPE PREPROCESSING
class Preprocessing(Enum):
    TOKENIZE = 1
    LEMMATIZE = 2
    LOWERCASE = 3
    REMOVE_STOP_WORDS = 4
    NUM = 5
    EMOTICONS = 6
    LINKS = 7
    EMAILS = 8
    PUNCT = 9
    REMOVE_ALPHA_NUMERIC = 10
    MENTIONS = 11
    HASHTAG = 12
    FIX_HASHTAG_TEXT = 13

# TYPE ENCODING
class EncodingMethod(Enum):
    BIGRAM = 1
    TRIGRAM = 2
    CUATRIGRAM = 3
    POSTAGGING = 4
    BERT = 5
    ALL_CHARGRAM = 6
    POS_ALL_CHARGRAM = 7

# TYPE CLASSIFICATION
class ClassificationMethod(Enum):
    SVM = 1
    SVM_RBC = 2
    PMSVM = 3
    BAYES = 4
    LOGISTIC_REGRESSION = 5
    BERT = 6

class TypeTask(Enum):
    VALIDATE_MODEL = 1
    VALIDATE_MODEL_BERT = 2
    PREDICTION = 3
    PREDICTION_BERT = 4
    MODEL_STORAGE = 5
    MODEL_STORAGE_BERT = 6
    PCA = 7