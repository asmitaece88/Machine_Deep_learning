# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:53:14 2019

@author: asmita chatterjee

Purpose:Keyword Extraction with TF-IDF and scikit-learn:
    
    We will use TF-IDF from the scikit-learn package to extract keywords from documents
    Keywords are simply descriptive words or phrases that characterize your document
    For example, keywords(topics) from this article would be tf-idf, 
     scikit-learn, keyword extraction, extract and so on.
     
data set source : stack overflow dump from Google’s Big Query.
 Type of data set : Bit noisy 
 
  2 data set files
  he larger file, stackoverflow-data-idf.json has 20,000 posts
  and is used to compute the Inverse Document Frequency (IDF) 
  and the smaller file, stackoverflow-test.json has 500 posts 
  and we would use that as a test set for us to extract keywords from
 """
 
 
import pandas as pd

#import numpy 
#import json 
# read json into a dataframe
df_idf=pd.read_json("C:/Users/USER/Documents/JSON/stackoverflow-data-idf.json",lines=True,orient='columns')
df_idf.head
# print schema
print("Schema:\n\n",df_idf.dtypes)

"""
accepted_answer_id          float64
answer_count                  int64
body                         object
comment_count                 int64
community_owned_date         object
creation_date                object
favorite_count              float64
id                            int64
last_activity_date           object
last_edit_date               object
last_editor_display_name     object
last_editor_user_id         float64
owner_display_name           object
owner_user_id               float64
post_type_id                  int64
score                         int64
tags                         object
title                        object
view_count                    int64
dtype: object

"""

print("Number of questions,columns=",df_idf.shape)
"""
Number of questions,columns= (20000, 19)
"""
"""
Notice that this stack overflow dataset contains 19 fields
What we are mostly interested in for this tutorial,
is the body and title which will become our source of text for keyword extraction

We will now create a field that combines both body and title 
so we have it in one field. We will also print the second text entry in our new field just to see what the text looks like.


2 demo JSOn field is :
    
    {"id":"3247246",
"title":"Integrate War-Plugin for m2eclipse into Eclipse Project",
"body":"\u003cp\u003eI set up a small web project with JSF and Maven. Now I want to deploy on a Tomcat server. Is there a possibility to automate that like a button in Eclipse that automatically deploys the project to Tomcat?\u003c/p\u003e\n\n\u003cp\u003eI read about a the \u003ca href=\"http://maven.apache.org/plugins/maven-war-plugin/\" rel=\"nofollow noreferrer\"\u003eMaven War Plugin\u003c/a\u003e but I couldn't find a tutorial how to integrate that into my process (eclipse/m2eclipse).\u003c/p\u003e\n\n\u003cp\u003eCan you link me to help or try to explain it. Thanks.\u003c/p\u003e",
"accepted_answer_id":"3247526",
"answer_count":"2",
"comment_count":"0",
"creation_date":"2010-07-14 14:39:48.053 UTC",
"last_activity_date":"2010-07-14 16:02:19.683 UTC",
"last_edit_date":"2010-07-14 15:56:37.803 UTC",
"last_editor_display_name":"",
"last_editor_user_id":"70604",
"owner_display_name":"",
"owner_user_id":"389430",
"post_type_id":"1",
"score":"2",
"tags":"eclipse|maven-2|tomcat|m2eclipse",
"view_count":"1653"}
"""


import re
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

df_idf['text'] = df_idf['title'] + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))
 
#show the 1st 'text' just for fun
df_idf['text'][0]

"""
Creating Vocabulary and Word Counts for IDF
IDf = total number of words in all documents / number of documents consisting of a
            particular word 
            
  For this  , need to create the vocabulary and start the counting process. 
  We can use the CountVectorizer to create a vocabulary from all the text 
  in our df_idf['text'] followed by the counts of words in the vocabulary.         
            
"""
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords 

 
#get the text column 
docs=df_idf['text'].tolist()
 
#create a vocabulary of words, 
"""ignore words that appear in 80% of documents, 
eliminate stop words"""
cv=CountVectorizer(max_df=0.8,stop_words=stopwords.words('english'))

word_count = cv.fit(docs)
word_count_vector=cv.fit_transform(docs)


print(cv.vocabulary_.keys())

"""OP of the print statemenet 
{'serializing': 97186, 'private': 84949, 'struct': 105977,
 'done': 31376, 'public': 86506, 'class': 18852, 'contains': 22286, 
 'properties': 85951, 'mostly': 69419, 'string': 105733, 'want': 120084,
 'serialize': 97155, 'attempt': 9311, 'stream': 105607, 'disk': 30187, 
 'using': 117533, 'xmlserializer': 123564, 'get': 43299, 'error': 35349, 
 'saying': 94824, 'types': 114501, 'serialized': 97161, 'need': 72631,
 'way': 120279, 'keep': 58989, 'prevent': 84546, 'floated': 40159, 
 ....................
"""


""" for small data sets ,we can transform a  term document matrix to array 
as below:
cv.transform(docs).toarray()

This would return  out of memory error for large data sets as our present data set 
"""
"""
The output of a CountVectorizer().fit_transform() is a sparse matrix.
 It means that it will only store the non-zero elements of a matrix.
 When you do print(X), only the non-zero entries are displayed as you 
 observe in the image
"""

"""
While cv.fit(...) would only create the vocabulary, cv.fit_transform(...) 
creates the vocabulary and returns a term-document matrix which is what we want. 
With this, each column in the matrix represents a word in the vocabulary while
 each row represents the document in our dataset where the values 
 in this case are the word counts. Note that with this representation, 
 counts of some words could be 0 if the word did not appear in the corresponding document.
"""
 
word_count_vector.shape
"""(20000, 125548)
this means:
    we have 20,000 documents in our dataset (the rows) 
    and the vocabulary size is 125548
"""
"""
let’s look at 15 words from our vocabulary.
"""
list(cv.vocabulary_.keys())[:15]

"""
We can also get the vocabulary by using get_feature_names()
"""

list(cv.get_feature_names())[2000:2015]

"""
TfidfTransformer to Compute Inverse Document Frequency (IDF)

ts now time to compute the IDF values. 
In the code below, we are essentially taking the sparse matrix from 
CountVectorizer (word_count_vector) to generate the IDF 
when you invoke tfidf_transformer.fit(...).


An extremely important point to note here is that the IDF should always be 
based on a large corpora and should be representative of texts you would be 
using to extract keywords. 
This is why we are using texts from 20,000 stack overflow posts to compute
 the IDF instead of just a handful
"""
from sklearn.feature_extraction.text import TfidfTransformer
 
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidftransform= tfidf_transformer.fit(word_count_vector)



"""Let's look at some of the IDF values:"""

tfidf_transformer.idf_

    
"""
Once we have our IDF computed, we are now ready to compute TF-IDF
 and then extract top keywords from the TF-IDF vectors. 
 
 In this example, we will extract top keywords for the questions in
 data/stackoverflow-data-idf.json
 
 
 So we invoke tfidf_transformer.transform(...)
 which  generates a vector of tf-idf scores.
 
 Next, we sort the words in the vector in descending order of tf-idf values 
and then iterate over to extract the top-n items with the corresponding feature names, 
In the example below, we are extracting keywords for the first document in our  data set.

The sort_coo(...) method essentially sorts the values in the vector while preserving the column index. 
Once you have the column index then its really easy to look-up the corresponding word value as you would see in 
extract_topn_from_vector(...) where we do feature_vals.append(feature_names[idx]).

"""

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)    
    print("",coo_matrix.col,"-","",coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
     fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
     score_vals.append(round(score, 3))
     feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
         results[feature_vals[idx]]=score_vals[idx]
    
    return results

# you only needs to do this once
feature_names=cv.get_feature_names()

# get the document that we want to extract keywords from

### getting the first document of the data set file 
doc=docs[0]

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
"""
{'contains': 0.188,
 'disk': 0.145,
 'mostly': 0.147,
 'private': 0.265,
 'public': 0.212,
 'serialize': 0.438,
 'serialized': 0.165,
 'serializing': 0.183,
 'struct': 0.632,
 'xmlserializer': 0.184}
"""

# now print the results
print("\n=====Title=====")
print(docs[0])
print("\n=====Body=====")
print(df_idf['body'].tolist)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])
    
    """
    
struct 0.632
serialize 0.438
private 0.265
public 0.212
contains 0.188
xmlserializer 0.184
serializing 0.183
serialized 0.165
mostly 0.147
disk 0.145
    
    """
    
  """  From the keywords above, the top keywords actually make sense,
    it talks about xmlserializer,disk are all unique to this specific question.
    
    There are many like contains , mostly etc ,, which can be fine tuned to remove
    if we decide to choose any custom stopwords list 
    we can  even create your own set of stop list,
    very specific to your domain.
    """
    
    """
    Check another record of the document data set 
    say check  the 50th record set 
    """
doc=docs[49]

tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,20)

# now print the results
print("\n=====Title=====")
print(docs[49])
"""
=====Title=====
how to find all wifi networks that are not in range i am writing an application to display the wifi network types and status how do i find all the not in range wifi networks is it possible to 
get the list of all configured previously seen wifi networks that are out of range i used the below code to get the result wifimanager mwifimanager wifimanager getsystemservice context wifi_service list 
lt wificonfiguration gt configs mwifimanager getconfigurednetworks list lt scanresult gt results mwifimanager getscanresults if configs null for wificonfiguration config configs for scanresult result 
results if result ssid null result ssid length continue else if result ssid equals mystring removedoublequotes config ssid int level mwifimanager calculatesignallevel result level log d myapp config ssid level 
but if configured network is high in number then it will take long time to execute is there any way to optimize this problem by getting the scanned result of only the configured network 
"""

print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])
    
    """
 ===Keywords===
ssid 0.425
mwifimanager 0.408
wifi 0.296
configs 0.25
result 0.248
networks 0.239
scanresult 0.204
wificonfiguration 0.196
wifimanager 0.186
configured 0.176
network 0.158
level 0.149
range 0.146
config 0.134
removedoublequotes 0.102
getscanresults 0.102
calculatesignallevel 0.102
list 0.099
getconfigurednetworks 0.098
wifi_service 0.095

    """
    
    
    """
    
     here also , we can fine tune  and remve the stop words
    
    """
    """ hence we can extract any set of records from 
         any record set 
         """