# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:06:14 2019

@author: Asmita Chatterjee

Data set source :http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

Purpose: classify wine type as poor , average or rich
"""
### Impor the packages 
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




## read the data directly from the data file 
data = pd.read_csv('C:/Users/USER/Documents/datasets/wine/winequality-red.csv',sep=';',quotechar='"')
data.head(4)
"""
 fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.4              0.70         0.00             1.9      0.076   
1            7.8              0.88         0.00             2.6      0.098   
2            7.8              0.76         0.04             2.3      0.092   
3           11.2              0.28         0.56             1.9      0.075   

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 11.0                  34.0   0.9978  3.51       0.56   
1                 25.0                  67.0   0.9968  3.20       0.68   
2                 15.0                  54.0   0.9970  3.26       0.65   
3                 17.0                  60.0   0.9980  3.16       0.58   

   alcohol  quality  
0      9.4        5  
1      9.8        5  
2      9.8        5  
3      9.8        6  
"""
### Check for missing values 


data.isnull().sum()
"""
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64

 no missing values 
"""
"""
target variable : quality :
    Finding out unique columns
"""

data['quality'].unique()
"""
 array([5, 6, 7, 4, 8, 3], dtype=int64)
 """
 
 data.columns
 """
Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'],
      dtype='object')
"""

data['quality'].value_counts() 
"""
5    681
6    638
7    199
4     53
8     18
3     10
Name: quality, dtype: int64
 so no missimg values . 
"""
"""
All wines with ratings less than 5 will fall under 0 (poor) category, 
wines with ratings 5 and 6 will be classified with the value 1 (average),
and wines with 7 and above will be of great quality (2).

"""

#Defining the splits for categories. 1–4 will be poor quality, 5–6 will be average, 6-8 will be great
bins = [1,4,6,8]
## we will have 3 intervales  so 4 bin values 


#0 for low quality, 1 for average, 2 for great quality
quality_labels=[0,1,2]

##3 add a new column quality_categorical
data['quality_categorical'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)

#Displays the first 10 records 
display(data.head(n=10))

"""
 fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.4              0.70         0.00             1.9      0.076   
1            7.8              0.88         0.00             2.6      0.098   
2            7.8              0.76         0.04             2.3      0.092   
3           11.2              0.28         0.56             1.9      0.075   
4            7.4              0.70         0.00             1.9      0.076   
5            7.4              0.66         0.00             1.8      0.075   
6            7.9              0.60         0.06             1.6      0.069   
7            7.3              0.65         0.00             1.2      0.065   
8            7.8              0.58         0.02             2.0      0.073   
9            7.5              0.50         0.36             6.1      0.071   

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 11.0                  34.0   0.9978  3.51       0.56   
1                 25.0                  67.0   0.9968  3.20       0.68   
2                 15.0                  54.0   0.9970  3.26       0.65   
3                 17.0                  60.0   0.9980  3.16       0.58   
4                 11.0                  34.0   0.9978  3.51       0.56   
5                 13.0                  40.0   0.9978  3.51       0.56   
6                 15.0                  59.0   0.9964  3.30       0.46   
7                 15.0                  21.0   0.9946  3.39       0.47   
8                  9.0                  18.0   0.9968  3.36       0.57   
9                 17.0                 102.0   0.9978  3.35       0.80   

   alcohol  quality quality_categorical  
0      9.4        5                   1  
1      9.8        5                   1  
2      9.8        5                   1  
3      9.8        6                   1  
4      9.4        5                   1  
5      9.4        5                   1  
6      9.4        5                   1  
7     10.0        7                   2  
8      9.5        7                   2  
9     10.5        5                   1  
"""

# Split the data into features and target label
quality_raw = data['quality_categorical']
features_raw = data.drop(['quality', 'quality_categorical'], axis = 1)

"""
quality_categorical will be treated as  the new target variable 
and the rest as features 
"""

## execute train tes splits 

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw, 
 quality_raw, 
 test_size = 0.3, 
 random_state = 1000)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
"""
Training set has 1119 samples.
"""
print("Testing set has {} samples.".format(X_test.shape[0]))
"""
Testing set has 480 samples.
"""

"""
we’ll run our training on an algorithm and evaluate its performance
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, model_selection
from sklearn.metrics import accuracy_score

classifier = RandomForestClassifier(max_depth = None  , random_state = None )

"""n_estimators" no of trees in the forest 
 max_fetures : The number of features to  consider , when looking for a split
 max_depth : the maximum depth  of the tree 

"""
rt_grid = {'n_estimators':[10,20,30,40,50],'max_depth':[3,4,5,None], 'max_features':[5,6,7,None]}


## Perform grid search on the classifier using the parameters rt_grid
grid_classifier = model_selection.GridSearchCV(classifier, rt_grid, cv=10, scoring='accuracy',refit=True, 
                                              n_jobs=-1, return_train_score=True)


## Perform  gird search object  on training data and find out optimal parameters 
grid_classifier.fit(X_train, y_train)

"""
GridSearchCV(cv=10, error_score='raise',
       estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False),
       fit_params=None, iid=True, n_jobs=-1,
       param_grid={'n_estimators': [10, 20, 30, 40, 50], 'max_depth': [3, 4, 5, None], 'max_features': [5, 6, 7, None]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
       scoring='accuracy', verbose=0)
       """

### get the best estimator 
final_model = grid_classifier.best_estimator_
 print(final_model)
 
 """
 RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=5, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
 """
 
 
  y_pred_final = final_model.predict(X_test)
  
  """
  array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
       1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2,
       1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
       1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2,
       1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
       1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
       1, 1, 0, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=int64)
"""
  
  
  print('Post optmization ')
  
   print("\n Final Accuracy score on testing data set {:4f}".format(accuracy_score(y_test,y_pred_final)))
   
   ## Final Accuracy score on testing data set 0.866667
   
   
   ## using the model  for predictions 
   
   """
   We can test our model to predict  model by giving it a bunch of  of values  for various features 
   and then check the prediction 
   
   
   Inputs given in the order :
       'fixed acidity' ,
       'volatile acidity', 
       'citric acid',
       'residual sugar',
       'chlorides',
       'free sulfur dioxide',
       'total sulfur dioxide',
       'density',
       'pH', 
       'sulphates',
       'alcohol'
   """
   
   
#####checking the feature importance 
   # Extract the feature importances using .feature_importances_ 
importances = final_model.feature_importances_

print(X_train.columns)

"""
Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol'],
      dtype='object')
"""
print(importances)
"""
[0.0768192  0.13051831 0.06872188 0.07990919 0.07327394 0.06160875
 0.08399519 0.07335585 0.06552744 0.12123555 0.16503472]
"""

