# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:20:00 2019

@author: USER
"""

## importing packages 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


docA = "The car is driven on the road"
docB = "The truck is driven on the highway"

bowA = docA.split(" ")
bowB = docB.split(" ")

## check each splitted  document / sentence 
bowA

bowB

## converting into a set and doing a  union gives us unique words 
wordSet = set(bowA).union(set(bowB))

wordSet
"""
Out[11]: {'The', 'car', 'driven', 'highway', 'is', 'on', 'road', 'the', 'truck'}
"""

""" converting the word set into a dicionary  consistig of word and keys
here sequence of elements as the keys of the dictionary.
"""
wordDictA = dict.fromkeys(wordSet, 0) 
wordDictB = dict.fromkeys(wordSet, 0)

""" now finding out the words in each document set ( sentence )
 and synching the count"""
for word in bowA:
    wordDictA[word]+=1
    
for word in bowB:
    wordDictB[word]+=1
    
    """
    if the word "car" is present in the sentence 1 then its count will be added     
    """
    
    wordDictA
    """
    {'The': 1,
 'car': 1,
 'driven': 1,
 'highway': 0,
 'is': 1,
 'on': 1,
 'road': 1,
 'the': 1,
 'truck': 0}      
    """
    
    wordDictB
    """
    {'The': 1,
 'car': 0,
 'driven': 1,
 'highway': 1,
 'is': 1,
 'on': 1,
 'road': 0,
 'the': 1,
 'truck': 1}
    """
    
    
import pandas as pd
pd.DataFrame([wordDictA, wordDictB])
"""
The  car  driven  highway  is  on  road  the  truck
0    1    1       1        0   1   1     1    1      0
1    1    0       1        1   1   1     0    1      1
"""

"""
wordDict.items give us ech word and its count in the dictionary 
"""
wordDictA.items()
"""
dict_items([('is', 1), ('driven', 1), ('car', 1), ('The', 1), ('highway', 0), 
('road', 1), ('the', 1), ('on', 1), ('truck', 0)])
"""


### compute TERM FREQUENCY --
def computeTF(wordDict, bow):
    tfDict = {}
    ### find out the total number of terms in the document 
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict


tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)

tfBowA
"""
{'The': 0.14285714285714285,
 'car': 0.14285714285714285,
 'driven': 0.14285714285714285,
 'highway': 0.0,
 'is': 0.14285714285714285,
 'on': 0.14285714285714285,
 'road': 0.14285714285714285,
 'the': 0.14285714285714285,
 'truck': 0.0}
"""

tfBowB
"""
{'The': 0.14285714285714285,
 'car': 0.0,
 'driven': 0.14285714285714285,
 'highway': 0.14285714285714285,
 'is': 0.14285714285714285,
 'on': 0.14285714285714285,
 'road': 0.0,
 'the': 0.14285714285714285,
 'truck': 0.14285714285714285}
"""

#### Now compute IDF 

def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict



"""
Pass both the dictionaries as a List , so as to find out N
"""

idfs = computeIDF([wordDictA, wordDictB])


idfs

"""
{'The': 0.0,
 'car': 0.3010299956639812,
 'driven': 0.0,
 'highway': 0.3010299956639812,
 'is': 0.0,
 'on': 0.0,
 'road': 0.3010299956639812,
 'the': 0.0,
 'truck': 0.3010299956639812}
"""


idfs[word]
"""
next compute TF-IDF 
"""

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf


"""
compute the tf-idf of each of the document sets 
"""
tfidfBowA = computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)


tfidfBowA
"""
{'The': 0.0,
 'car': 0.043004285094854454,
 'driven': 0.0,
 'highway': 0.0,
 'is': 0.0,
 'on': 0.0,
 'road': 0.043004285094854454,
 'the': 0.0,
 'truck': 0.0}
"""

tfidfBowB
"""
{'The': 0.0,
 'car': 0.0,
 'driven': 0.0,
 'highway': 0.043004285094854454,
 'is': 0.0,
 'on': 0.0,
 'road': 0.0,
 'the': 0.0,
 'truck': 0.043004285094854454}
"""


import pandas as pd
pd.DataFrame([tfidfBowA, tfidfBowB])

"""
 The       car  driven   highway   is   on      road  the     truck
0  0.0  0.043004     0.0  0.000000  0.0  0.0  0.043004  0.0  0.000000
1  0.0  0.000000     0.0  0.043004  0.0  0.0  0.000000  0.0  0.043004

hence car and truck are the 2 words , which have the highest TF-IDF
"""