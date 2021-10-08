# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score

# #Python
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

# import os
# import sys

# FILE = __file__
# DATA_SET_FOLDER = os.path.split(FILE)[0]
# TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
# PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

# sys.path.append(PROJECT_FOLDER)

# from tweetlib.data_set.data_set import DataSet
# from tweetlib.config.configuration import Configuration
# from tweetlib.encoding import postagging
# from tweetlib.definitions import TaggingMethod

# class SVM(object):
    
#     # classify
#     #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#     def run_model_balanced(X_train, X_test, y_train, y_test):
#         clf = SVC(kernel='linear', C=1.0,random_state=1,class_weight="balanced")
#         clf.fit(X_train, y_train)
#         return clf
    
#     #ejecutamos el modelo "tal cual"
#     model = run_model_balanced(X_train, X_test, y_train, y_test)

#     #Realizo una predicción
#     y_pred = model.predict(X_test)

#     #definimos funciona para mostrar los resultados
#     def mostrar_resultados(y_test, y_pred):
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         plt.figure(figsize=(12, 12))
#         LABELS= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
#         plt.title("Confusion matrix")
#         plt.ylabel('True class')
#         plt.xlabel('Predicted class')
#         plt.show()
#         print (classification_report(y_test, y_pred))

#     mostrar_resultados(y_test, y_pred)

#     # print(f"y_train: {y_train}")
#     # print(f"y_test: {y_test}")
#     # # print(f"X_test: {X_pred}")
#     # print(f"y_pred: {y_pred}")

#     # #Verifico la matriz de Confusión
#     # matriz = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     # print('Matriz de Confusión:')
#     # print(matriz)

#     # #Calculo la precisión del modelo
#     # print("Precision Score : ",precision_score(y_test,y_pred,pos_label='positive', average='micro'))
#     # print("Recall Score :" , recall_score(y_test, y_pred, pos_label='positive', average='micro'))
#     # # print('Precisión del modelo:')
#     # # print(precision)