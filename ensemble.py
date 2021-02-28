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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


X = np.load("data/X.npy", allow_pickle=True)
y = np.load("data/y.npy", allow_pickle=True)

dX_train, dX_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(dX_train)
print(y_train)

X_train = []
X_test = []

dX_test = dX_test.reshape(-1,1)

X_train = np.array(dX_train)
X_test = np.array(dX_test)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)
classNames = np.unique(y)

hit = 0
total = 0


print(accuracy_score(y_test, y_pred))
