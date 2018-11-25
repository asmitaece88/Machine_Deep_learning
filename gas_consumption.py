# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:26:55 2018

@author: USER
"""

###gas consumption

"""
We will use this dataset to try and predict gas consumptions (in millions of gallons) 
in 48 US states based upon gas tax (in cents), per capita income (dollars),
 paved highways (in miles) and the proportion of population with a drivers license.
 
data source :
     http://people.sc.fsu.edu/~jburkardt/datasets/regression/x16.txt

"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline

dataset = pd.read_csv('petrol_consumption.csv')  

dataset.head() 

X = dataset.drop('Petrol_Consumption', axis=1)  
y = dataset['Petrol_Consumption']  

## divide te data set ---
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train)  


y_pred = regressor.predict(X_test)  

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
df  

##now evaluate the algorithm 

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

sum_total=dataset['Petrol_Consumption'].sum()
##27685

## Since mean abslute error is 61.33 . which is < 10 % of um_total , hence this model is fine 