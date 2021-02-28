import pandas as pd
import numpy as np## for plotting
import matplotlib.pyplot as plt
import seaborn as sns## for processing
import re
import nltk## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing## for explainer
from lime import lime_text## for word embedding
import gensim
import gensim.downloader as gensim_api## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K## for bert language model
import transformers
import preProcess as pp
from sklearn.model_selection import train_test_split
from sklearn import feature_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X = np.load("data/X.npy", allow_pickle=True)
y = np.load("data/y.npy", allow_pickle=True)




dX_train, dX_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(dX_train)
print(y_train)

dX_train = dX_train.reshape(-1,1)
X_train = []
X_test = []

dX_test = dX_test.reshape(-1,1)
clf = DecisionTreeClassifier(random_state=0)
for val in dX_train:
    X_train.append(val)
    if not(float('-inf') < float(val) < float('inf')):
        X_train.append(0)
        
for val in dX_test:
    X_train.append(val)
    if not(float('-inf') < float(val) < float('inf')):
        X_test.append(0)
        

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = np.array([np.nan, 1, 2])




X_train = X_train.reshape(-1,1)
X_test = X_train.reshape(-1, 1)


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
classNames = y.unique()

print(classification_report(y_test, y_pred, target_names=classNames))

