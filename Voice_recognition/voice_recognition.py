# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:54:20 2019

@author: asmita chatterjee

 data source :https://www.kaggle.com/primaryobjects/voicegender/version/1
 
 Purpose :To classify between male and female voice 
 
"""
import seaborn as sns
##from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split, learning_curve, StratifiedShuffleSplit, GridSearchCV,
    cross_val_score)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

%matplotlib inline

## read the dataframe 
voice_df  = pd.read_csv( "C:/Users/USER/Documents/datasets/voice.csv",encoding="utf8" )

### next check for ny null columns 
voice_df.isnull().sum()
"""
meanfreq    0
sd          0
median      0
Q25         0
Q75         0
IQR         0
skew        0
kurt        0
sp.ent      0
sfm         0
mode        0
centroid    0
meanfun     0
minfun      0
maxfun      0
meandom     0
mindom      0
maxdom      0
dfrange     0
modindx     0
label       0
dtype: int64

"""

### hence there are no columns with null values

voice_df.shape
###(3168, 21)

### Next check the balance of the dataset 
print("Number of male: ",(voice_df[voice_df.label == 'male'].shape[0]))
##
"""Number of male:  1584
print("Number of female: .format(df[df.label == 'female'].shape[0]))
"""


print("Number of female: ",(voice_df[voice_df.label == 'female'].shape[0]))
##
"""Number of female:  1584
print("Number of female: ".format(df[df.label == 'female'].shape[0]))
"""


### so the dataset is exactly balanced 

## get the lables  only in a dataframe 
from sklearn.preprocessing import LabelEncoder
y=voice_df.iloc[:,-1]

# Encode label category
# male -> 1
# female -> 0

gender_encoder = LabelEncoder()

## converting the labels into numeric values using labelencodr r
y = gender_encoder.fit_transform(y)
y

### get all the features except the label column in a dataframe X 
X = voice_df.loc[:, voice_df.columns != 'label']

X.shape
###(3168, 20)
"""
3168 is the number of rows , and 20 is the number of columns 
"""
X.head(2)
"""
meanfreq        sd    median       Q25       Q75       IQR       skew  \
0  0.059781  0.064241  0.032027  0.015071  0.090193  0.075122  12.863462   
1  0.066009  0.067310  0.040229  0.019414  0.092666  0.073252  22.423285   

         kurt    sp.ent       sfm  mode  centroid   meanfun    minfun  \
0  274.402906  0.893369  0.491918   0.0  0.059781  0.084279  0.015702   
1  634.613855  0.892193  0.513724   0.0  0.066009  0.107937  0.015826   

     maxfun   meandom    mindom    maxdom   dfrange   modindx  
0  0.275862  0.007812  0.007812  0.007812  0.000000  0.000000  
1  0.250000  0.009014  0.007812  0.054688  0.046875  0.052632  
"""
### next find out the correlation of the input variable paarameters 
"""
### from the above plot , we find the parameters skew and kurt are aving high rage of values as compared to other columns
## hence standardisation of the input variables are   mandatory . but before that check the correlation  of input variables
"""
voice_df.corr()

## Plot the correlation 
correlation = voice_df.corr()
plt.figure(figsize=(30,30))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')

plt.title('Correlation between different features')

##  from the plot we find   there are quite a few paramters , which are  highly crrelated  like correlation value is 0.634
## as we see there are quite a few variable pairs , which have high correlation like meanfreq and Q25 as well as sfm and sp.ent
## hence it is necessary to apply PCA to reduce   the pairs which do have apprppriate variance  
## since svm is an expensive algorithm in  terms of computational complexity 
## first apply standardisation of scaling , since scaling is a mandatory action  before applying PCA

# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


X_scaled.shape##(3168, 20)

X_scaled

"""
array([[-4.04924806,  0.4273553 , -4.22490077, ..., -1.43142165,
        -1.41913712, -1.45477229],
       [-3.84105325,  0.6116695 , -3.99929342, ..., -1.41810716,
        -1.4058184 , -1.01410294],
       [-3.46306647,  1.60384791, -4.09585052, ..., -1.42920257,
        -1.41691733, -1.06534356],
       ...,
       [-1.29877326,  2.32272355, -0.05197279, ..., -0.5992661 ,
        -0.58671739,  0.17588664],
       [-1.2452018 ,  2.012196  , -0.01772849, ..., -0.41286326,
        -0.40025537,  1.14916112],
       [-0.51474626,  2.14765111, -0.07087873, ..., -1.27608595,
        -1.2637521 ,  1.47567886]])
"""
##  since there are 30 parameters . le us apply PCA and make the number of fatures to 4 using PCA 
### now apply PCA 
from sklearn.decomposition import PCA
pca = PCA()
x_pca=pca.fit_transform(X_scaled)


"""
array([[ 8.20851631e+00,  2.16448836e+00,  1.95978393e+00, ...,
        -8.22248773e-15,  4.70587730e-15,  1.65014481e-15],
       [ 8.67189184e+00,  3.85462661e+00,  4.10720799e+00, ...,
        -8.06563024e-15,  1.16816761e-15, -1.58035136e-16],
       [ 9.11116887e+00,  4.51914027e+00,  7.52825158e+00, ...,
         7.84641481e-15,  1.04107928e-15,  3.46567144e-17],
       ...,
       [ 3.83137839e+00, -1.78143380e+00,  1.87075519e-01, ...,
         3.33495099e-15,  2.67095300e-16,  1.62146806e-17],
       [ 3.21441858e+00, -1.95563325e+00, -8.00871476e-01, ...,
        -9.33017780e-15, -8.05047550e-17,  4.29533564e-17],
       [ 2.36104138e+00, -1.33959575e+00, -9.13831984e-01, ...,
        -4.13700980e-15,  1.60020428e-16,  1.18687324e-17]])
"""

explained_variance=pca.explained_variance_ratio_
explained_variance

"""
array([4.52163908e-01, 1.18706090e-01, 1.09099393e-01, 7.61976317e-02,
       5.29393771e-02, 4.61496635e-02, 3.20448218e-02, 2.89839393e-02,
       2.45172645e-02, 1.87551752e-02, 1.65590573e-02, 8.95842514e-03,
       6.90291504e-03, 4.69046383e-03, 2.28912851e-03, 6.45523808e-04,
       3.97221935e-04, 1.92581584e-30, 3.95056363e-33, 7.51705998e-35])
"""

type(explained_variance)##numpy.ndarray  20

 
 x_pca.shape## (3168, 4)

## next we plot a scree plot

"""
The first thig we do is calculate  the percentage variation that each  principal component accounts for

"""  
per_var = np.round (explained_variance*100 , decimals=1) 


"""
next we create labels for the scree plot  there are like Pc1 , Pc2 , Pc 3 , etc (1 label per Pc)
"""
labels = ['PC'+str(x) for x  in range(1,len(per_var)+1)]

"""
next we use a matplotlib to create a bar 
"""

with plt.style.context('dark_background'):
    plt.figure(figsize=(8,8))
    plt.bar(x=range(1,len(per_var)+1),height = per_var, tick_label = labels)
    plt.ylabel('% of Explained variance ',fontsize =7)
    plt.xlabel('Principal components',fontsize =7)
    plt.title("Scree plot")
    plt.show()
"""
from the plot , it is clear that  variances of Principal components 
aftre Pc11 redduce drsatically 

Hence we remove the PC(s) aftre Pc11
"""
pca=PCA(n_components=11)
X_pca_new=pca.fit_transform(X_scaled)
X_pca_new

"""
array([[ 8.20851631,  2.16448836,  1.95978393, ..., -1.81514737,
        -1.12503499,  0.31278609],
       [ 8.67189184,  3.85462661,  4.10720799, ..., -0.39054506,
        -1.80502479, -0.65686317],
       [ 9.11116887,  4.51914027,  7.52825158, ..., -0.45755386,
        -2.17865538, -0.41993239],
       ...,
       [ 3.83137839, -1.7814338 ,  0.18707552, ...,  1.2580664 ,
         1.67608928,  1.07359808],
       [ 3.21441858, -1.95563325, -0.80087148, ...,  1.60568365,
        -0.76911144,  1.10872144],
       [ 2.36104138, -1.33959575, -0.91383198, ...,  1.12367223,
        -0.68914858,  1.08095587]])
    """
    
  X_pca_new.shape  ##(3168, 11)
  
  """
Splitting dataset into training set and testing set for better generalisation
"""
## note : here y is label encoded 
##X_pca_new 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca_new, y, test_size=0.3, random_state=100)


"""
now we will process SVM  stepwise , 

by applying single train test split , 
then applying CV using grdisearch 
and finnaly calculaating the accuracy report 

 each for 
 
 -->linear 
 --> rbf
 ---> polynomial

"""

"""
For linear kernel 
"""

"""
single train test split
"""
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

## check the accuracy score 

print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))##0.9705573080967402

print( 'Report : ',classification_report(y_test, y_pred) )
"""
print( 'Report : ',classification_report(y_test, y_pred) )
Report :               precision    recall  f1-score   support

          0       0.97      0.97      0.97       471
          1       0.97      0.97      0.97       480

avg / total       0.97      0.97      0.97       951
"""

confusion_matrix_df = pd.DataFrame(
    metrics.confusion_matrix(y_test, y_pred),
    index=[['actual', 'actual'], ['male', 'female']],
    columns=[['predicted', 'predicted'], ['male', 'female']]
)


confusion_matrix_df

"""

 predicted       
                   male female
actual male         456     15
       female        13    467
       
         if the classfication use case is to check whether the  voice is male or not 
         
         the here , 
         
         Tp : 456 
         FN: 15
         FP: 13
         TN:467
       """
  """
The C parameter tells the SVM optimization how much you want to avoid misclassifying
 each training example. For large values of C, 
 the optimization will choose a smaller-margin hyperplane  
 if that hyperplane does a better job of getting all the training points classified correctly.
 Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating
 hyperplane, even if that hyperplane misclassifies more points.
 hence we run the k fold cross valdiatiom  for various values of C
"""     

## mention a range of C_range for getting the optimum values first  on the entire data set 
C_range=list(range(1,5))
acc_score=[]


for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())## accumulate the accuracy scores for linear kernel 
print(acc_score)    
"""
[0.919321710054932, 0.9464334049836797, 0.9587035267892683, 0.9624830825571214]
"""


## plot the accuracy score--star from here 

import matplotlib.pyplot as plt
%matplotlib inline


C_values=list(range(1,5))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0,10,0.5))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')

"""
as checked from the Fig3 , the CV score is highest at around 4.0 
"""

### now lets perform GridsearchCV on the series of C values ending at c=4.0

from sklearn.svm import SVC
svm_model= SVC()

grid_param = {
 'C': (np.arange(0.1,4,0.1)) , 'kernel': ['linear']
                   }


from sklearn.grid_search import GridSearchCV

grid_search_model = GridSearchCV(svm_model, grid_param,cv=10,scoring='accuracy', n_jobs=-1)

## fit the model with the svm model run by gridsearchCV , into trainung and testing daata set 
grid_search_model.fit(X_train, y_train)
print(grid_search_model.best_score_)##0.97744700045106

print(grid_search_model.best_params_)
"""{'C': 1.0, 'kernel': 'linear'}
"""


y_pred= grid_search_model.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))

##0.9705573080967402

"""
hence , we observe that using  gridsearchCV , the accuracy  core   of train and test dat sets are 
nearly equal

accuracy score of  training data set  is : 0.97744700045106

accuracy score of  training data set  is : 0.9705573080967402



 hence , the  moel using linear kernel and hyperparmatar C as 1.0  gives us good results 
 
 We havent used kernel polynmial or rbf , since using those require a lot of time of execution.
"""
