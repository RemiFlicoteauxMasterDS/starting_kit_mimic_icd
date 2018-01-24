
from __future__ import unicode_literals
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pandas as pd
import numpy as np



class Classifier():
    def __init__(self):
        #self.raw_embedding = load_embedding_from_url(url='http://nlp.stanford.edu/data/glove.6B.zip', filename='glove.6B.200d.txt')
        self.clf = RandomForestClassifier()
        # self.metaclf = XGBClassifier()

    def fit(self, X, y):

        self.clf.fit(X, y)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y = np.argmax(y_proba, axis=1)
        return y

    def predict_proba(self, X):
        y_proba = self.clf.predict_proba(X)
        
        return  y_proba