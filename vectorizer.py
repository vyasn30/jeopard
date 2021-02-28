import pandas as pd                                           
from gensim.models.keyedvectors import KeyedVectors           
from gensim.models import Word2Vec                            
import numpy as np                                            
import matplotlib.pyplot as plt                               
import spacy                                                  
import unidecode                                              
from word2number import w2n                                   
import contractions                                           
from bs4 import BeautifulSoup                                 
from tqdm import tqdm                                         
import gensim                                                 
from gensim import corpora                                    
from pprint import pprint                                     
from gensim.utils import simple_preprocess                    
from gensim.models import Word2Vec                            
from gensim.models.keyedvectors import KeyedVectors           
from nltk.stem import PorterStemmer                           
from sklearn.feature_extraction.text import TfidfVectorizer 
from statistics import mean

def getWeight():
    print("fuck you")
    Questions = []
    X = []

    with open('data/listfile.txt', 'r') as filehandle:
        for line in filehandle:
            currentQuestion = line[:-1]
            Questions.append(currentQuestion)





    vectorizer = TfidfVectorizer(min_df=1)
    lols = vectorizer.fit_transform(Questions)
    idf = vectorizer.idf_


    weightDict = dict(zip(vectorizer.get_feature_names(), idf))

    for sen in tqdm(Questions):                                 
        alnum = ""
        vector = []
        for character in sen:

            if character.isalnum():
                alnum += character
            if not(character.isalnum()):
                alnum += " "
    
        sen = alnum.split()
        for word in sen:
            try:
                vector.append(weightDict[word])
        
            except Exception:
                vector.append(0)
        try:
            X.append(mean(vector))
        except Exception:
            X.append(0)


    return X

if __name__ == "__main__":
    getWeight()




