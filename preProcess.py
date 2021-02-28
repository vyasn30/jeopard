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



nlp = spacy.load("en_core_web_sm")

deselect_stop_words = ['no', 'not']



for w in deselect_stop_words:
  nlp.vocab[w].is_stop = False
  
 
def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text

def text_preprocessing(text, accented_chars=True, contractions=True, 
                       convert_num=True, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=True):
    """preprocess text with default option set to true for all steps"""
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    if contractions == True: #expand contractions
        text = expand_contractions(text)
    if lowercase == True: #convert all characters to lowercase
        text = text.lower()

    doc = nlp(text) #tokenise text

    clean_text = []
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)
  
    return clean_text

    

def toNum(value):
  value = value.replace(",", "")
  value = value.replace("$", "")
  return int(value)  

def binning(value):
  if value < 1000:
    return np.round(value , -2)
  
  elif value < 10000:
    return np.round(value, -3)

  else:
    return np.round(value, -4)


def main():
  model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)


  df = pd.read_csv("data/JEOPARDY_CSV.csv")
  df = df[df[' Value'] != 'None']
  
  #making a new column of value without "$" and ","

  df['ValueNum'] = df[' Value'].apply(
  toNum
  )

#too many classes lets bin it

  df['ValueBins'] = df['ValueNum'].apply(binning)
  
  df.drop([" Value", "ValueNum"], axis=1, inplace = True)
 
  
  features = list(df.columns.values)

  classLabels = df['ValueBins'].unique()
  ans = df['ValueBins']
  features = list(df.columns.values)
  



  y = []

  Questions = []
  X = []

  with open('data/listfile.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentQuestion = line[:-1]

        # add item to the list
        Questions.append(currentQuestion)


  for i, sen in tqdm(enumerate(Questions)):
    vector = []
    
    for word in sen:
        try:
            vector.append(model[word])
        except Exception:
            continue
    vector = np.array(vector)
    X.append(vector.mean())
 

  y = np.array(ans)
  X = np.array(X)
  X = np.nan_to_num(X)


  for val in X:
      print(val)

  print(X.shape)
  print(y.shape)
  np.save("data/y.npy", y)
  np.save("data/X.npy", X)
    
      

if __name__ == "__main__":
        main()
