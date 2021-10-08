import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from tweetlib import TaggingMethod
from tweetlib import SpanishTextSamples
from tweetlib.encoding import postagging_encoding 

def compute_X():
    list_texts = SpanishTextSamples.get_all_texts()
    vectors = []
    for i in list_texts:
        v = postagging_encoding(i, TaggingMethod.STANZA)
        vectors.append(v)
    X = np.vstack(vectors)
    return X

def compute_y(X):
    y_length = X.shape[0]
    y = np.zeros(y_length, dtype=int)
    y[:y_length // 2] = 1
    np.random.shuffle(y)
    return y

def run():
    X_train = compute_X()
    y_train = compute_y(X_train)
    X_test = X_train

    #ajusta el modelo
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)


    print(f"y_train: {y_train}")
    print(f"y_test: {y_test}")

if __name__ == "__main__":
    run()