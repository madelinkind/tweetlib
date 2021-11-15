import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score

#Python
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.metrics import classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
# from sklearn.tree import DecisionTreeClassifier

# from pylab import rcParams

# from imblearn.under_sampling import NearMiss
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.combine import SMOTETomek
# from imblearn.ensemble import BalancedBaggingClassifier

import os
import sys

FILE = __file__
DATA_SET_FOLDER = os.path.split(FILE)[0]
TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

sys.path.append(PROJECT_FOLDER)

from tweetlib.data_set.data_set import DataSet
from tweetlib.config.configuration import Configuration
from tweetlib.classification.classification import Classification
# from tweetlib.preprocessing.remove_stop_words import Stop_words
from tweetlib.definitions import TypeTask
# from tweetlib.init_nlp import init_nlp
# from tweetlib.singleton import NlpFactory

from tweetlib.preprocessing.prep_definitions import dict_preprocessing
from tweetlib.encoding.enc_definitions import dict_encoding
from tweetlib.classification.task_definition import dict_task 
from tweetlib.singleton import Utils 
# from typing import Optional
from joblib import dump

class TwitterPipeline(object):

    def __init__(self, config: Configuration, classifier: Classification = None, task: TypeTask = None, dataset: DataSet = None, text: list = None, id_model: str = None, n_value: int = None, model = None, len_labels = None):
        super(TwitterPipeline, self).__init__()

        self.config = config
        self.dataset = dataset
        self.classifier = classifier
        self.task = task
        self.text = text
        self.id_model = id_model
        self.n_value = n_value
        self.model = model
        self.len_labels = len_labels

    def run(self):

        data_sents = []
        y_label = []

        # get data and classes from self.data
        if self.dataset != None:
            data = self.dataset.get_data()
            y = self.dataset.get_y()
            #copy to data
            data_texts = data.copy()
        if self.config.get_tagging_method() != None:
            tagging_method_type = self.config.get_tagging_method()
            #INITIALIZE THE LIBRARY TO USE
            nlp = Utils.load_nlp(tagging_method_type)
        
        preprocessing_list = self.config.get_preprocessing_methods()
        encoding = self.config.get_encoding_method()
        classifier_type = self.config.get_classification_method()
        # type_user = self.config.get_type_user()
        # vectors = []
        if self.task != None:
            # gettting the type of task
            type_task = dict_task[self.task]

            if self.task.name == 'PREDICTION' or self.task.name == 'PREDICTION_BERT' and self.dataset == None:
                if len(self.text) != 0: 
                    data_texts = self.text
                    y = None
                else:
                    print('The file is empty, you must enter a file with one or more texts to predict')
                    return
            elif self.dataset == None:
                print('You must enter a data set both to store the model and to validate it')
                return

        for indx, preprocessing in enumerate(preprocessing_list):
            print(f"Comenzando preprocesamiento '{preprocessing.name}'...")
            prep_method = dict_preprocessing[preprocessing]
            if preprocessing.name != 'LOWERCASE' and preprocessing.name != 'REMOVE_STOP_WORDS' and preprocessing.name != 'MENTIONS':
                for idx, text_prep in enumerate(data_texts):
                    prep = prep_method(text_prep)
                    data_texts[idx] = prep
            else:
                for idx, text in enumerate(data_texts):
                    prep = prep_method(text)
                    data_texts[idx] = prep
            if len(preprocessing_list)-1 == indx: 
                print("Preprocesamiento completado.")

        #Verificar que luego del procesamiento no queden listas vacías
        for idx, sent in enumerate(data_texts):
            if sent != []:
                data_sents.append(sent)
                if y != None:
                    y_label.append(y[idx])
                else:
                    y_label = None
            else:
                continue

        encoding_method = dict_encoding[encoding]
        print(f"Comenzando la extracción de características '{encoding.name}'...")
        #Para obtener los datos para bert preprocesados
        # if encoding.name == 'BERT':
        #     dictionary_X_y = {'X': data_sents, 'y': y} 
        #     dump(dictionary_X_y, f'BERT_DATA_SET/1000/politic_bert_token_stopword_lowercase_lemma')

        if  encoding.name != 'BERT':
            if encoding.name == 'BIGRAM' or encoding.name == 'TRIGRAM' or encoding.name == 'CUATRIGRAM':
                vector_encoding = encoding_method(data_sents)
            elif encoding.name == 'ALL_CHARGRAM':
                vector_encoding = encoding_method(data_sents)
            elif encoding.name == 'POS_ALL_CHARGRAM':
                vector_encoding = encoding_method(data_sents, tagging_method_type.name, nlp)
            else:
                vector_encoding = encoding_method(data_sents, tagging_method_type.name, nlp)
            X = np.vstack(vector_encoding)
            nan = np.isnan(X)
            X[nan] = 0.0

        else:
            vector_encoding_bert = encoding_method(data_sents, y_label)
        print("Extracción completada.")

        if self.task.name == 'VALIDATE_MODEL':
            accuracy, recall, f1, total_class, total_tweets = type_task(X, y_label, classifier_type)
            return accuracy, recall, f1, total_class, total_tweets

        elif self.task.name == 'VALIDATE_MODEL_BERT':
            accuracy, recall, f1, total_class, total_tweets = type_task(vector_encoding_bert[0], vector_encoding_bert[1], vector_encoding_bert[2], vector_encoding_bert[4], vector_encoding_bert[5])
            return accuracy, recall, f1, total_class, total_tweets

        elif self.task.name == 'PREDICTION':
            type_task(self.model, X, self.n_value)

        elif self.task.name == 'PREDICTION_BERT':
            type_task(self.model, vector_encoding_bert[0], vector_encoding_bert[1], self.n_value, self.len_labels)
        
        elif self.task.name == 'MODEL_STORAGE':
            type_task(self.id_model, self.config, X, y_label, classifier_type)
        
        elif self.task.name == 'MODEL_STORAGE_BERT':
            type_task(self.id_model, self.config, classifier_type, vector_encoding_bert[0], vector_encoding_bert[1], vector_encoding_bert[2], vector_encoding_bert[3], vector_encoding_bert[4])

        #PCA
        else:
            type_task(X, y_label)