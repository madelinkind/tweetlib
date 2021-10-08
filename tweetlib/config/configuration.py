import os
import sys

FILE = __file__
DATA_SET_FOLDER = os.path.split(FILE)[0]
TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

sys.path.append(PROJECT_FOLDER)

#Posiblemente estas importaciones no sean necesarias aqui, eliminar luego
from tweetlib.definitions import Preprocessing, EncodingMethod, ClassificationMethod, TaggingMethod

class Configuration(object):

    def __init__(self, preprocessing_methods: list = [], encoding_method: int = None, classification_method: int = None, tagging_method: int = None, type_dataset: str = None):
        self.preprocessing_methods = preprocessing_methods
        self.encoding_method = encoding_method
        self.classification_method = classification_method
        self.tagging_method = tagging_method
        # self.type_task = type_task
        self.type_dataset = type_dataset

    def get_preprocessing_methods(self) -> list: 
        return self.preprocessing_methods

    def set_preprocessing_methods(self, preprocessing_methods: list):
        self.preprocessing_methods = preprocessing_methods
    
    def get_encoding_method(self) -> int:
        return self.encoding_method
    
    def set_encoding_method(self, encoding_method: int):
        self.encoding_method = encoding_method

    def get_classification_method(self) -> int:
        return self.classification_method
    
    def set_classification_method(self, classification_method: int):
        self.classification_method = classification_method
    
    def get_tagging_method(self) -> int:
        return self.tagging_method

    def set_tagging_method(self, tagging_method: int):
        self.tagging_method = tagging_method

    # def get_type_task(self) -> int:
    #     return self.type_task
    
    # def set_type_task(self, self.type_task: int):
    #     self.type_task = type_task

    def get_type_dataset(self) -> str:
        return self.type_dataset

    def set_type_dataset(self, type_dataset: str):
        self.type_dataset = type_dataset
    # list_of_preprocessing_to_apply = [
    #     Preprocessing.CLEAN_TWEET,
    #     Preprocessing.CLEAN_EMOTICONS,
    #     Preprocessing.LEMMATIZE,
    #     Preprocessing.STOP_WORDS
    # ]
# clase que se encarga del preprocesamiento
    # p = TwitterPipeline(c1)