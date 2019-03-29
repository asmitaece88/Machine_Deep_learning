# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:37:46 2019

@author: asmita chatterjee
"""

"""
Purpose:Text mining on Wine reviews data set 

Dataset source : https://www.kaggle.com/zynicide/wine-reviews/version/4
Using csv file : winemag-data-130k-v2

Algoirithm used is :  TF-IDF
"""


import numpy as np
import pandas as pd
import csv 


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
###nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

##read the  the data csv file 
file = open("winemag-data-130k-v2.csv", encoding="utf8")
reader = csv.reader(file)

### remove the header 

"""
each row is a wine observation and each column is a variable.
 Of all the variables—location, province, region, 
pricing, grape variety, etc.—we are most interested in description, the column of wine reviews
"""

"""
Next, we’d like to extract the reviews. (For ease of comprehension,
 we will use the colloquial words “word” as opposed to “term,”
  “review” as opposed to “document”). 
  reviews are in column 3 of the dataset
"""

"""
each review is a string. I
"""

"""
First convert the revews string to tokenize byusing .split() and .replace()
"""

data = [
    [(word.replace(",", "")
          .replace(".", "")
          .replace("(", "")
          .replace(")", ""))
    for word in row[2].lower().split()]
    for row in reader]
    
  ### remove the header     
data = data[1:]

data[0]

"""
since the variable data  is a List consisting of Lists , 
so we cannot directly pass the  data as a corpus to the Tfidvectorizer.fittransform
Hence we manually find out tf and idf , and finally tfidf values 
"""

""" first compute a TF map

= # of times the term appears in document / total # of terms in document 
Now that our data is usable, we’d like to start computing the TF and the IDF. 
Recall that computing the tf of a word in a document requires us to calculate the number
 of words in a review, and the number of times each word appears in the review. 
 We can store each (word, word count pair) in a dictionary.
 The keys of the dictionary are then just the unique terms in the review. 
The following function takes in a review and outputs a tf dictionary for that review.
"""
def ReviewTFDict(review):
    """ Returns a tf dictionary for each review whose keys are all 
    the unique words in the review and whose values are their 
    corresponding tf.
    """
    #Counts the number of times the word appears in review
    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1
    #Computes tf for each word           
    for word in reviewTFDict:
        reviewTFDict[word] = reviewTFDict[word] / len(review)
    return reviewTFDict

    
"""
We then call our function on every review and store all these tf dictionaries. 
We do this using a list comprehension, where each index maps to a review’s tf dictionary

declare a Tfdictionary list variable 
"""

tfDict = {}
i=0

"""
 using a for loop , get the Tf values of all the  review records 
"""
for i in range(len(data)):
    tfDict[i] = ReviewTFDict(data[i])
    
    
## checking the TF df value ofach word in the 1st review
tfDict[0]
"""
{'acidity': 0.041666666666666664,
 'alongside': 0.041666666666666664,
 'and': 0.08333333333333333,
 'apple': 0.041666666666666664,
 'aromas': 0.041666666666666664,
 'brimstone': 0.041666666666666664,
 'brisk': 0.041666666666666664,
 'broom': 0.041666666666666664,
 'citrus': 0.041666666666666664,
 'dried': 0.08333333333333333,
 'expressive': 0.041666666666666664,
 'fruit': 0.041666666666666664,
 'herb': 0.041666666666666664,
 'include': 0.041666666666666664,
 "isn't": 0.041666666666666664,
 'offering': 0.041666666666666664,
 'overly': 0.041666666666666664,
 'palate': 0.041666666666666664,
 'sage': 0.041666666666666664,
 'the': 0.041666666666666664,
 'tropical': 0.041666666666666664,
 'unripened': 0.041666666666666664}
"""


"""
Computing an IDF Map
IDF(term) = log(total # of documents / # of documents with term in it) 

   Computing the idf of a word requires us to compute the total number of documents
   and the number of documents that contains the word.
   In our case we can calculate the total number of documents with len(data), the number of wine reviews. 
   For each review, we increment the document count for each unique word.
   We can use the keys of the dictionaries that we calculated in the TF step to 
   get the unique set of words. The resulting IDF dictionary’s keys will be the set
   of all unique words across every document.
"""

data 
def computeCountDict():
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countDict = {}
    
    #countDict's (word, doc) pair
    ### check each review from data     
    for review in data:
        for word in review:
            ### below code will insert only unique words in
            ## countDict()
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict

  #Store the review count dictionary
countDict = computeCountDict()

### testing by checking a word from countdictionary

countDict["pairing"]
##Out[69]: 464

"""
this means pairing is used in  464  records or documents  in the entire corpus( in all the 13k records)
"""


"""
Finally, we can compute an idfDict, using countDict and some math, and store it.
"""
def computeIDFDict():
    import math
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(len(data) / countDict[word])
    return idfDict
  
 #Stores the idf dictionary
idfDict = computeIDFDict()

idfDict["pairing"]
"""
 5.635182075403029
 """
 
 
"""
 Computing the TF-IDF Map
 
 TF-IDF(term) = TF(term) * IDF(term) 
The last step is to compute the TF-IDF.
 We use our existing tf dictionaries and simply multiply each value by the idf. 
We can use the idf keys since they contain every unique word.
"""


def computeTFIDF(tfBow, idfDict):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfDict[word]
    return tfidf

tfidf={}

  #Stores the TF-IDF dictionaries
for i in range(len(data)):
   tfidf[i] = computeTFIDF(tfDict[i],idfDict)
   
   tfidf[0]
   
   """
   {'acidity': 0.054744975310773894,
 'alongside': 0.132607518840734,
 'and': -0.08192973912866552,
 'apple': 0.0965224555791442,
 'aromas': 0.04954008931276635,
 'brimstone': 0.3160588285667858,
 'brisk': 0.1714956358318011,
 'broom': 0.2713490182801833,
 'citrus': 0.10157622176936462,
 'dried': 0.23369836325785845,
 'expressive': 0.2227227978995895,
 'fruit': 0.04428732607757831,
 'herb': 0.1258579300560466,
 'include': 0.21942906211073346,
 "isn't": 0.2217267345306914,
 'offering': 0.16440975375357053,
 'overly': 0.22894461619454543,
 'palate': 0.051204572167602005,
 'sage': 0.17963041763168167,
 'the': -0.022106172478552417,
 'tropical': 0.14906366199617085,
 'unripened': 0.4448522641233823}
   """
   
   
###Constructing A Vector

"""Now that we have our TF-IDF dictionaries, we can create our matrix.
 Our matrix will be an array of vectors, where each vector represents a review.
 The vector will be a list of frequencies for each unique word in the dataset—the TF-IDF 
 value if the word is in the review, or 0.0 otherwise.
 """
 
# Create a list of unique words
wordDict = sorted(countDict.keys())

def computeTFIDFVector(review):
      tfidfVector = [0.0] * len(wordDict)
     
      # For each unique word, if it is in the review, store its TF-IDF value.
      for i, word in enumerate(wordDict):
          if word in review:
              tfidfVector[i] = review[word]
      return tfidfVector

for review  in tfidf:
     tfidfvector = computeTFIDFVector(review)