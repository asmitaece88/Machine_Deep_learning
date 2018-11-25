# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:00:21 2018

@author: USER
"""
"""
we will predict whether a bank note is authentic or fake depending upon the
 four different attributes of the image of the note.
 The attributes are Variance of wavelet transformed image, 
curtosis of the image, entropy, and skewness of the image
"""


## source of data set 
##https://archive.ics.uci.edu/ml/datasets/banknote+authentication

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline


dataset = pd.read_csv("bill_authentication.csv")  
dataset.shape
dataset.head()

## drop the target variable to get the iput predictor variables in a dataframe 

X= dataset.drop('Class', axis=1)

y = dataset['Class']

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) 


from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)


"""
Now that our classifier has been trained, let's make predictions on the test data. 
To make predictions, the predict method of the DecisionTreeClassifier class is used.
"""


y_pred = classifier.predict(X_test)

##Evaluating the Algorithm

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score 

print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
print("Report : ", 
    classification_report(y_test, y_pred)) 