# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:26:21 2018

@author: USER
"""

import pandas as pd
df = pd.read_csv("salaries.csv")
df.head()
data = df.drop('salary_more_then_100k',axis=1)

data

target = df['salary_more_then_100k']
target

from sklearn.preprocessing import LabelEncoder
###from sklearn.preprocessing import OneHotEncoder

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

data['company_n'] = le_company.fit_transform(data['company'])
data['job_n'] = le_company.fit_transform(data['job'])
data['degree_n'] = le_company.fit_transform(data['degree'])

data_new = data.drop(['company','job','degree'],axis=1)

data_new

""""onehotencoder_0 = OneHotEncoder(categorical_features = [0])
onehotencoder_1= OneHotEncoder(categorical_features = [1])
onehotencoder_2= OneHotEncoder(categorical_features = [2])
data_new = onehotencoder.fit_transform(data_new).toarray()
"""

target

from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
dt_model = DecisionTreeClassifier()


clf_object=dt_model.fit(data_new, target)

target_pred = clf_object.predict(data_new) 

confusion_matrix(target, target_pred)
      
   
accuracy_score(target,target_pred)*100
      

       
