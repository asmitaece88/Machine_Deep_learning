# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:37:46 2019

@author: asmita chatterjee
"""

"""
Purpose:Text mining on Wine reviews data set 

Dataset source : https://www.kaggle.com
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
   
   tfidf[1]
    
     """
     {'2016': 0.11380015345836172,
 'a': -0.008490552453230629,
 'acidity': 0.034575773880488776,
 'already': 0.11001803644940705,
 'although': 0.10298261973171852,
 'and': -0.05174509839705191,
 'are': 0.04251325725115718,
 'be': 0.0758938893950697,
 'berry': 0.05599544952777387,
 'better': 0.11550567235537042,
 'certainly': 0.1306269012902103,
 'drinkable': 0.13178219743062874,
 'filled': 0.1566321245301649,
 'firm': 0.06981608343475457,
 'freshened': 0.2016892832488414,
 'from': 0.036416516428888615,
 'fruits': 0.05965775914074526,
 'fruity': 0.06895087534012738,
 'is': 0.015564138364476482,
 'it': 0.02232560623354938,
 "it's": 0.03867597393083618,
 'juicy': 0.06817525362314754,
 'out': 0.07446325461714788,
 'red': 0.05086219022153037,
 'ripe': 0.04137701835596421,
 'smooth': 0.07672529323974082,
 'still': 0.08180504561834981,
 'structured': 0.08486383110965387,
 'tannins': 0.03787401088914437,
 'that': 0.033215631858908995,
 'this': 0.0034510320102208534,
 'while': 0.06326233886249968,
 'will': 0.07301860053792186,
 'wine': 0.013477763381385875,
 'with': 0.004024573638911964}
     
     """
   
###Constructing A Vector

"""
suppose , we want to see first 2 records of review in a matrix form 
we will load everything into pandas dataframe 
"""

import pandas as pd
pd.DataFrame([tfidf[0], tfidf[1]])


""" 
2016         a   acidity  alongside   already  although       and  \
0     NaN       NaN  0.054745   0.132608       NaN       NaN -0.081930   
1  0.1138 -0.008491  0.034576        NaN  0.110018  0.102983 -0.051745   

      apple       are   aromas    ...      tannins      that       the  \
0  0.096522       NaN  0.04954    ...          NaN       NaN -0.022106   
1       NaN  0.042513      NaN    ...     0.037874  0.033216       NaN   

       this  tropical  unripened     while      will      wine      with  
0       NaN  0.149064   0.444852       NaN       NaN       NaN       NaN  
1  0.003451       NaN        NaN  0.063262  0.073019  0.013478  0.004025  
"""
