# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:48:50 2020

@author: eksalkeld
"""
from constants import *
from pandas.io.json import json_normalize
import pandas as pd
import json
import nltk
#nltk.download('stopwords')

#JSON_FILE='C://Users//eksalkeld//Documents//GitHub//data-scientist-exercise02//data//NarrativeData_000.json'
#with open(JSON_FILE, 'r') as f:
#    jsondata = json.load(f)
#jsondf=json_normalize(jsondata['data'])


import os
txtdf = pd.DataFrame()
read_in=[file for file in os.listdir(JSON_FOLDER) if file.endswith('.json')]
for file in read_in:
    full_filename = "%s/%s" % (JSON_FOLDER, file)
    with open(full_filename,'r') as fi:
        jsondata = json.load(fi)
        jsondf=json_normalize(jsondata['data'])
        txtdf=txtdf.append(jsondf)

import gensim
import re
import string

txtdf=txtdf.reset_index(drop=True)

#compile all the punctuation marks, which will be stripped out
punc = re.compile( '[%s]' % re.escape( string.punctuation ) )

def remove_punc(data):
    
    #All text to lower case
    data = data.lower()
    #Punctuation subbed out for ''
    data = punc.sub( '', data)
    
    return data

#all lower case and remove the puncuation
#txtdf['nopunc']=txtdf.apply(lambda x: punc.sub('',x['probable_cause'].lower()), axis=1)
txtdf['nopunc']=txtdf.apply(lambda x: remove_punc(x['probable_cause']),axis=1)



#Tokenize the words in each string
txtdf['wordtoken']=txtdf.apply(lambda x: nltk.word_tokenize(x['nopunc']), axis=1)

#Define the stop words to strip out from the term vectors
stop_words = nltk.corpus.stopwords.words( 'english' )


def remove_stop(data,stop_words):
    
    #Ensure the word isn't in the list of defined stop words
    clean_list=[x for x in data if x not in stop_words]
    
    return clean_list

txtdf['nostop']=txtdf.apply(lambda x: remove_stop(x['wordtoken'],stop_words), axis=1)

#Stem the words down to their root    
# Porter stemmer initalization
porter=nltk.stem.porter.PorterStemmer()
#Snowball stemmer
snowball=nltk.stem.SnowballStemmer("english")

def apply_stem(data,stemmer):
    
    #Loop through each word in the vector
    for i in range(len(data)):
        #Apply the stemmer and replace the original value with the new value
        data[i]=stemmer.stem(data[i])
        
    return data

#Apply the stemmer to the words in each row
txtdf['stem']=txtdf.apply(lambda x : apply_stem(x['nostop'],porter), axis=1)
    

#Alternatively could lemmatize
wordnetlem=nltk.stem.WordNetLemmatizer()
def apply_lemm(data,lemm):
    
    #Loop through each word in the vector
    for i in range(len(data)):
        #Apply the stemmer and replace the original value with the new value
        data[i]=lemm.lemmatize(data[i])
        
    return data




from collections import defaultdict

def sum_word_inst(data):
    
    #Dictionary, word as key, instance count as value
    term_ct = defaultdict(int)
    #Loop through each of the words
    for i in data:
        #Increase sum of times the word has been seen thus far by 1
        term_ct[i] += 1
        
    return term_ct
#For each row (description of accident/event) find how often each word is used
txtdf['word_ct']=txtdf.apply(lambda x: sum_word_inst(x['stem']), axis=1)

#texts = [
#    [token for token in text if term_ct[token] > 1]
 #   for text in texts
#]

#txtdf.apply(lambda x: gensim.corpora.Dictionary(x['stem'].split() ),axis=1)
#Dictionary of corpus
wdict=gensim.corpora.Dictionary(txtdf['stem'])
#ID map for words across the entire corpus
wdict.token2id
#How many docs (rows) a word comes up in
for i in range(len(wdict)):
    print(wdict[i]+": "+str(wdict.dfs[i]))
    
corpus_w_ct=[]
for i in range(len(wdict)):
    corpus_w_ct.append([wdict[i],wdict.dfs[i]])
pd.DataFrame(corpus_w_ct,columns=['word','count']).sort_values('count')

#Use the id map for the entire corpus to create a bag of words for each document
#word replaced by the ID
#returns list of (token_id, token_count) tuples
txtdf['BOW']=txtdf.apply(lambda x: wdict.doc2bow(x['stem']),axis=1)

#Create the TFIDF vectors for the entire corpus
tfidfmodel=gensim.models.TfidfModel(list(txtdf['BOW']))

#Apply TFIDF on the documents
#List of tuples with dictionary ID and the new weight
txtdf['tfidf']=txtdf.apply(lambda x: tfidfmodel[x['BOW']],axis=1)

#Nuber of unique words
uniquewords = len( wdict )
#corp is the BOW
simindex = gensim.similarities.SparseMatrixSimilarity( txtdf['tfidf'], num_features = uniquewords )
#Number of docs by number of docs similarity matrix
txtdf.apply(lambda x: simindex[x['BOW']],axis=1)

