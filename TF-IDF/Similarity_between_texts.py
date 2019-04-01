# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:02:00 2019

@author: Asmita Chatterjee
"""
"""
Purpose : To find out the similarity of 2 text messages 
In text analysis, each vector can represent a document. The greater the value of θ,
 the less the value of cos θ, thus the less the similarity between two documents.
 
 we have 4 documents as 4 sentences . Need to find the similarity
 
 The calculated tf-idf is normalized by the Euclidean norm so that
 each row vector has a length of 1. 
 The normalized tf-idf matrix should be in the shape of n by m.
 A cosine similarity matrix (n by n)
 can be obtained by multiplying the if-idf matrix by its transpose (m by n).

"""

d1 = "plot: two teen couples go to a church party, drink and then drive."
d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . "
d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling . "
d4 = "damn that y2k bug . "
documents = [d1, d2, d3, d4]


"""
preprocessing  with nltk
"""

##Normalize by lemmatization:
import nltk, string, numpy
nltk.download('punkt') 
nltk.download('wordnet') 


"""
we want meaningful terms in dictionary terms 
hence we perform lemmatization
"""

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]
 
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
 
def LemNormalize(text):
 return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

"""Turn text into vectors of term frequency:
    """
from sklearn.feature_extraction.text import CountVectorizer
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')

##fit_transform makes  Term frequency  matrix 
tf_matrix =  LemVectorizer.fit_transform(documents).toarray()

"""
array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
      dtype=int64)
"""
tf_matrix.shape
"""Out[86]: (4, 41)
"""
    
"""
This should be a 4 (# of documents) by 41 (# of terms in the corpus). Check its shape:
"""


"""Normalized (after lemmatization) text in the four documents are tokenized 
and each term is indexed:
    """
    
print(LemVectorizer.vocabulary_)
"""
{'plot': 27, 'teen': 37, 'couple': 9, 'church': 6, 'party': 25, 
'drink': 14, 'drive': 15, 'film': 17, 'adapted': 0, 'comic': 8, 'book': 3, 
'plenty': 26, 'success': 32, 'theyre': 38, 'superheroes': 33, 'batman': 2, 
'superman': 34, 'spawn': 29, 'geared': 18, 'kid': 22, 'casper': 5, 'arthouse': 1,
 'crowd': 11, 'ghost': 19, 'world': 39, 'really': 28, 'like': 23, 'hell': 20,
 'movie': 24, 'come': 7, 'suspect': 36, 'studio': 31, 'indication': 21, 
 'stinker': 30, 'everybodys': 16, 'surprise': 35,
 'critical': 10, 'darling': 13, 'damn': 12, 'y2k': 40, 'bug': 4}
"""

print(LemVectorizer.get_feature_names())
""" this gives the features or the vocaulary or list of words
['adapted', 'arthouse', 'batman', 'book', 'bug', 'casper', 'church', 
'come', 'comic', 'couple', 'critical', 'crowd', 'damn', 'darling', 'drink', 
'drive', 'everybodys', 'film', 'geared', 'ghost', 'hell', 'indication', 'kid',
 'like', 'movie', 'party', 'plenty', 'plot', 'really', 'spawn', 'stinker', 'studio', 
 'success', 'superheroes', 'superman', 'surprise', 'suspect', 'teen', 'theyre',
 'world', 'y2k']
"""

"""Calculate idf and turn tf matrix to tf-idf matrix:
    """
from sklearn.feature_extraction.text import TfidfTransformer
tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)
print( tfidfTran.idf_)

"""
[1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073
 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073
 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 1.51082562
 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073
 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073
 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073
 1.91629073 1.91629073 1.91629073 1.91629073 1.91629073]
"""

"""
Now we have a vector where each component is the idf for each term. 
In this case, the values are almost the same because other than one term, 
each term only appears in 1 document. The exception 
is the 18th term that appears in 2 document
"""

"""Now 
Get the tf-idf matrix (4 by 41):
    """
    
tfidf_matrix = tfidfTran.transform(tf_matrix)
print(tfidf_matrix.toarray())

"""
[[0.         0.         0.         0.         0.         0.
  0.37796447 0.         0.         0.37796447 0.         0.
  0.         0.         0.37796447 0.37796447 0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.37796447 0.         0.37796447 0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.37796447 0.         0.         0.        ]
 [0.19381304 0.19381304 0.19381304 0.38762607 0.         0.19381304
  0.         0.         0.38762607 0.         0.         0.19381304
  0.         0.         0.         0.         0.         0.15280442
  0.19381304 0.19381304 0.19381304 0.         0.19381304 0.19381304
  0.         0.         0.19381304 0.         0.19381304 0.19381304
  0.         0.         0.19381304 0.19381304 0.19381304 0.
  0.         0.         0.19381304 0.19381304 0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.27094807 0.         0.         0.27094807 0.
  0.         0.27094807 0.         0.         0.27094807 0.21361857
  0.         0.         0.         0.27094807 0.         0.
  0.27094807 0.         0.         0.         0.         0.
  0.27094807 0.54189613 0.         0.         0.         0.27094807
  0.27094807 0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.57735027 0.
  0.         0.         0.         0.         0.         0.
  0.57735027 0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.57735027]]
 """
 
 
 """
 Here what the transform method does is multiplying the tf matrix (4 by 41) 
 by the diagonal idf matrix (41 by 41 with idf for each term on the main diagonal), 
 and dividing the tf-idf by the Euclidean norm. 
 
 """
 """
 Now Get the pairwise similarity matrix (n by n):
     """
     
cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print(cos_similarity_matrix)

"""
array([[1.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.03264186, 0.        ],
       [0.        , 0.03264186, 1.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        ]])
     """
     
"""
 to get similarity  of the documnts , we  multiply the matrix obtained in the last step is multiplied by its transpose
 
 
 The result is the similarity matrix, which indicates that d2 and d3 are more similar 
 to each other than any other pair.
 """
 
 
 