# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:15:21 2019

@author: asmita chatterjee

data set source :https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
 
 The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. 
 It contains one set of SMS messages in English of 5,571 messages, tagged acording being ham (legitimate) or spam.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

import seaborn as sns
##from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, svm
from sklearn.model_selection import (
    train_test_split, learning_curve, StratifiedShuffleSplit, GridSearchCV,
    cross_val_score)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 



###load the csv file 
mails_df  = pd.read_csv( "C:/Users/USER/Documents/datasets/spam.csv",encoding="latin-1" )

mails_df.head(2)
"""
 v1                                                 v2 Unnamed: 2  \
0  ham  Go until jurong point, crazy.. Available only ...        NaN   
1  ham                      Ok lar... Joking wif u oni...        NaN   

  Unnamed: 3 Unnamed: 4  
0        NaN        NaN  
1        NaN        NaN  
"""
"""
since the  1st column denotes the target variable containing class labels 
deniting the sms message is spam or ham 
Now check out the distribution of the 2 class labels 
"""
mails_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 5 columns):
v1            5572 non-null object
v2            5572 non-null object
Unnamed: 2    50 non-null object
Unnamed: 3    12 non-null object
Unnamed: 4    6 non-null object
dtypes: object(5)
memory usage: 217.7+ KB
"""

### Inspecting dataset 
# To check the email type as spam or not spam(ham) we dont need the Unnamed columns. so we drop the same
mails_df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
mails_df.head(3)
"""
 v1                                                 v2
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
"""

# rename the v1 and v2 columns as labels and message respectively
mails_df.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
mails_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 2 columns):
labels     5572 non-null object
message    5572 non-null object
dtypes: object(2)
memory usage: 87.1+ KB
"""


# based on whether the  label value is ham ( no spam) or spam  we mention a  filter column and turn it as 0 or 1 
mails_df['label'] = mails_df['labels'].map({'ham': 0, 'spam': 1})
mails_df.head(4)


## get the counts 

mails_df['labels'].value_counts()
"""
ham     4825
spam     747
Name: labels, dtype: int64

It looks like there are far fewer training examples 
for spam than ham—we'll take this imbalance into account in the analysis
"""

### store the target variable label in a different dataframe
target = mails_df['label']

# Encode the class labels as numbers
le = LabelEncoder()
target_enc = le.fit_transform(target)

### get the raw text n a variable 

rawtext = mails_df['message']

mails_df.head(3)
"""
labels                                            message  label
0    ham  Go until jurong point, crazy.. Available only ...      0
1    ham                      Ok lar... Joking wif u oni...      0
2   spam  Free entry in 2 a wkly comp to win FA Cup fina...      1
"""


### text prcessing 

ENGLISH_STOPWORDS = set(stopwords.words('english'))

porter = nltk.PorterStemmer()

## define a function , which preprocss and removes or replaces special charactares 
def preprocess_text(input_string):
    assert(type(input_string) == str)
    ## change all email id(s)  to emailid 
    clean = re.sub(r'([\w\-.]+?@\w+?\.\w{0,9}\b)', 'emailid', input_string)
    
    ## change the 
    re.sub( r'(https[s]?\S+)|(\w+\.[A-Za-z]{0,9})',"URL",clean)
    
    ## change the  currency symbols   to currency 
    clean = re.sub(r'£|\$', 'Currency', clean)
    
    clean = re.sub(r'\d+(\.\d+)?', 'NUMBER', clean)
   
    clean=clean.lower()
    
     # remove special characters and digits with number
    ###clean=re.sub("(\\d|\\W)+","number",clean)
    
    clean = ' '.join(word for word in clean.split() if word not in ENGLISH_STOPWORDS) 
    
    return ' '.join(
      porter.stem(term) 
      for term in clean.split()
    )
    
    
    
mails_df['message'] = mails_df['message'].apply(preprocess_text)

mails_df.head(4)
"""
labels                                            message  label
0    ham  go jurong point, crazy.. avail bugi n great wo...      0
1    ham                        ok lar... joke wif u oni...      0
2   spam  free entri number wkli comp win fa cup final t...      1
3    ham          u dun say earli hor... u c alreadi say...      0
"""


"""
lets take a hypothestical sms  message 

Please CALL 08712402578 immediately as there is an urgent message waiting for you
"""

testing = "Please CALL 08712402578 immediately as there is an urgent message waiting for you"

preprocess_text(testing)



## next is feature engineering 

"""
We can tokenize individual terms and generate what's called a bag of words model. 
But this desnt capture the innate fatures 
Under this model, the following sentences have the same feature vector although they convey dramatically different meanings.

Is she not a good girl?
She is a good girl.

 we will be using bigram( n=2) model  to prevent this problem 
 
 o get the best of both worlds, let's tokenize unigrams and bigrams. As an example, unigrams and bigrams for
 "The quick brown fox" are "The", "quick", "brown", "fox", "The quick", "quick brown" and "brown fox"
"""


## next stepis TFDF vectorizer

"""
Having selected a tokenization strategy, the next step is assign each $n$-gram to 
a feature and then compute the $n$-gram's frequency using some statistic

we will do this using tf-idf statistic

"""

# Construct a design matrix using an n-gram model and a tf-idf statistics


### first assign  the message of the DF  to a variable named cleaned_message


cleaned_msg = mails_df['message']

cleaned_msg.head(3)
"""
0    go jurong point, crazy.. avail bugi n great wo...
1                          ok lar... joke wif u oni...
2    free entri number wkli comp win fa cup final t...
Name: message, dtype: object
"""

"""
we're equipped to transform a corpus of text data into a matrix of numbers with one row per training example 
and one column per $n$-gram
"""
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_ngrams = vectorizer.fit_transform(cleaned_msg)

## lets look at the dimensions  of the X_ngrams matrx 

X_ngrams.shape

## (5572, 39920)

## hence  a matrix of 5572 rows and 39920 columns

"""
hence the tokenization process extracted total 39920 unigrams as well as bigrams 

 So X_ngrams consist of all the tf-idf values of all the  unigram as well as bigrams generated aftre TFIDFvectorizer
 X_ngrams is a sparse matrix
 
"""
### feature enginering completed 

###Training and evaluating a model
"""
since now  we have preprocessed  as well as done feature engineering 
we can call SVM classifier - by which we can find the hyperplane which separates the two classificatio  criteria 
- spam and ham 

 we would ne using a linear kernel for SVM , and not a non linear kernel , since that is computationally expensive
"""

###prepare the training and testing data set 

X_train, X_test, y_train, y_test = train_test_split(
    X_ngrams,
    target_enc,
    test_size=0.3,
    random_state=100,
    stratify=target_enc
)

# Train SVM with a linear kernel on the training set
clf = svm.LinearSVC(loss='hinge')
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set

## this determines a single train-test split 
y_pred = clf.predict(X_test)

# Since the data is imbalanced that is 
## there are unequal counts of postive ( spam) and negative ( ham)
## messages so we apply F1 score as an evaluation metric 
###Compute the F1 score
metrics.f1_score(y_test, y_pred)
##0.9267139479905437

###let us no build a confusion matrix to take a peek at what types of mistakes the classifier is making 


confusion_matrix_df = pd.DataFrame(
    metrics.confusion_matrix(y_test, y_pred),
    index=[['actual', 'actual'], ['spam', 'ham']],
    columns=[['predicted', 'predicted'], ['spam', 'ham']]
)


confusion_matrix_df
"""
                 predicted     
                 spam  ham
actual spam      1445    3
       ham         28  196
       """
"""
here we see the numbr of false postive are quite high that is 28 
that is marking spam messages as Ham 
"""
###printing other paameters 
       
print('Accuracy Score :',accuracy_score(y_test, y_pred) )
###0.9814593301435407   not bad!!!
print( 'Report : ',classification_report(y_test, y_pred) )
       
"""
precision    recall  f1-score   support

          0       0.98      1.00      0.99      1448
          1       0.98      0.88      0.93       224

avg / total       0.98      0.98      0.98      1672
"""

"""
This is the resut of a singe train/test spplit 
"""

### to check  whether the model is overfitted  and suffers from high variance ,
### let us apply gridsearch cross validation to find out the best hyperparameter 
### for our model 
### hyperparameter for us is C , since we have already considered that we will be using linear kernel
"""
Using nested cross-validation, let's test a range of 20 values for the regularization hyperparameter 
and use 10-fold cross-validation to assess the classifier's performance.

"""
## determine the  parameers to be passed in gridsearchCV-- range of 20 values 
param_grid = [{'C': np.logspace(-4, 4, 20)}]


## call the gridsearch CV   
grid_search = GridSearchCV(
    estimator=svm.LinearSVC(loss='hinge'),
    param_grid=param_grid,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42),
    scoring='f1',
    n_jobs=-1
)

scores = cross_val_score(
    estimator=grid_search,
    X=X_ngrams,## pass the input matrix 
    y=target_enc,## pass the label encoded target variable 
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0),
    scoring='f1',
    n_jobs=-1
)

scores

"""
array([0.90556901, 0.94444444, 0.93647059, 0.92857143, 0.91904762,
       0.95348837, 0.91566265, 0.93617021, 0.9342723 , 0.93735499])
    
    """
scores.mean()###vhence the mean cross valication score is found to be 0.9311051611892655

### run from the below code 

### Now we need to use the optimal regularization hyperparameter to train
##the classifier on the whole dataset in order to provide it as much information as possible


grid_search.fit(X_ngrams, target_enc)


# View the accuracy score
print('Best score for data1:', grid_search.best_score_) 
"""
Best score for data1: 0.9267583793233674
"""

print('Best C:',grid_search.best_estimator_.C) 
"""
1.623776739188721
"""

###This tells us that the most accurate model uses C=1.623776739188721


## now apply the best hyperparameter C on the emtre data set 
final_clf = svm.LinearSVC(loss='hinge', C=1.623776739188721)
final_clf.fit(X_ngrams,target_enc);

    ##############  
   def spam_filter(message):
    if final_clf.predict(vectorizer.transform([preprocess_text(message)])):
        return 'spam'
    else:
        return 'not spam'
    
    
    
    ## testing    
    


    spam_filter('Ohhh, but those are the best kind of foods')
###'not spam'
    
    
        example = """  ***** CONGRATlations **** You won 2 tIckETs to Hamilton in 
NYC http://www.hamiltonbroadway.com/J?NaIOl/event    !! !  """
    
    spam_filter(example)
    
    ##'spam'
    
    
    example2= "Congrats ********* you have won 400!!http://www.amazon.com $!!"
    
       spam_filter(example2)
       
       ###'spam'