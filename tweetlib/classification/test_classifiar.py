import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from pylab import rcParams

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier

import os
import sys

FILE = __file__
DATA_SET_FOLDER = os.path.split(FILE)[0]
TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

sys.path.append(PROJECT_FOLDER)

from tweetlib.data_set.data_set import DataSet
from tweetlib.config.configuration import Configuration
from tweetlib.encoding import postagging
from tweetlib.definitions import TaggingMethod, ClassificationMethod
# from tweetlib.pipeline.execute_pipeline import TwitterPipeline

class Classification(object):

    # def __init__(self, data: TwitterPipeline):
    #     super(Classification, self).__init__()
    accuracy=[]
    # classify
    def classification_method(self, X, y, method: ClassificationMethod):
       #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
        #dividimos en sets de entrenamiento y test
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        skf = StratifiedKFold(n_splits=5, random_state=None)
        skf.get_n_splits(X,y)
        StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
        for train_index, test_index in skf.split(X, y):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #ejecutamos el modelo
            model = self.run_model_balanced(X_train, X_test, y_train, y_test, method)
            # se realiza las predicciones en los datos de prueba usando predict()
            pred_y = model.predict(X_test)
            score = accuracy_score(pred_y, y_test)
            accuracy.append(score)
        print(accuracy)
        print(np.array(accuracy).mean())

        # show_results(y_test, pred_y)

    def run_model_balanced(self, X_train, X_test, y_train, y_test, method: ClassificationMethod):
        #clasificador a utilizar
        if method == ClassificationMethod.LOGISTIC_REGRESSION:
            clf = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
        if method == ClassificationMethod.SVM:
            clf = SVC(kernel='linear',class_weight="balanced")
        # if method == ClassificationMethod.BAYES:
        #     clf = MultinomialNB(class_weight="balanced")
        #Ajustamos los datos de entrenamiento en el clasificador usando fit(). Entrenar nuestro modelo
        clf.fit(X_train, y_train)
        return clf