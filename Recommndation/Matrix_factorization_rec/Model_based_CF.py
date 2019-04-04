# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:18:50 2019

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:48:02 2019

@author: asmita chatterjee

Source of dataset :
    https://grouplens.org/datasets/movielens/
    dataset name : ml-1m.zip 
   
This rich and rare dataset contains a real sample of 12 months logs (Mar. 2016 - Feb. 2017) 
from CI&T's Internal Communication platform (DeskDrop). 
I contains about 73k logged users interactions on more than 
3k public articles shared in the platform.  
    
"""
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


"""
Latent factor models compress user-item matrix into a low-dimensional representation 
in terms of latent factors. One advantage of using this approach is that instead of having 
a high dimensional matrix containing abundant number of missing values 
we will be dealing with a much smaller matrix in lower-dimensional space.
A reduced presentation could be utilized for either user-based or item-based neighborhood algorithms
 that are presented in the previous section. 
 There are several advantages with this paradigm. 
 It handles the sparsity of the original matrix better than memory based ones.
 Also comparing similarity on the resulting matrix is much more scalable especially
 in dealing with large sparse datasets.
 
 we will be using SVD
"""

###Setting Up the Ratings Data

ratings_list = [i.strip().split("::") for i in open('C:/Users/USER/Documents/datasets/ml-1m/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('C:/Users/USER/Documents/datasets/ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('C:/Users/USER/Documents/datasets/ml-1m/movies.dat', 'r').readlines()]


""" convert the data sets into arrays 
"""
ratings = np.array(ratings_list)
users = np.array(users_list)
movies = np.array(movies_list)

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
ratings_df.head(3)
"""
UserID  MovieID  Rating  Timestamp
0       1     1193       5  978300760
1       1      661       3  978302109
2       1      914       3  978301968
"""

movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])

movies_df.dtypes
"""
MovieID    object
Title      object
Genres     object


 convert the MOvie id to numeric 
"""

movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)    


movies_df.head(3)

"""
MovieID                    Title                        Genres
0        1         Toy Story (1995)   Animation|Children's|Comedy
1        2           Jumanji (1995)  Adventure|Children's|Fantasy
2        3  Grumpier Old Men (1995)                Comedy|Romance
"""

"""Now  from the above format of ratings df , We need to format the  ratings matrix
 to be one row per user and one column per movie. 
pivot ratings_df to get that and call the new variable Ratings_new_df
"""
ratings_new_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
ratings_new_df.head(3)
"""
MovieID  1     2     3     4     5     6     7     8     9     10    ...   \
UserID                                                               ...    
1         5.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  ...    
2         0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  ...    
3         0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  ...    

MovieID  3943  3944  3945  3946  3947  3948  3949  3950  3951  3952  
UserID                                                               
1         0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  
2         0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  
3         0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
"""

"""
 next , normalize tne rating     data  by each user's  mean , 
 since raing can be very high ot low as pr user's choice.
 Next , convert the data  from a dataframe to a numpy array. 
 """
 
ratings_new_df_matrix = ratings_new_df.as_matrix()
 
ratings_new_df_matrix.head(4)

user_ratings_mean = np.mean(ratings_new_df_matrix, axis = 1)
"""
axis 1 means  finding mean across rows 
"""
ratings_new_df_norm = ratings_new_df_matrix - user_ratings_mean.reshape(-1, 1)

### next step is SVD 

from scipy.sparse.linalg import svds

U, sigma, Vt = svds(ratings_new_df_norm, k = 40)

"""
here sgma doesnt give us a diagonal mtrix , hence we convert it to diagonal matrix form
"""
sigma = np.diag(sigma)
"""
array([[ 158.55444246,    0.        ,    0.        , ...,    0.        ,
           0.        ,    0.        ],
       [   0.        ,  159.49830789,    0.        , ...,    0.        ,
           0.        ,    0.        ],
       [   0.        ,    0.        ,  161.17474208, ...,    0.        ,
           0.        ,    0.        ],
       ...,
       [   0.        ,    0.        ,    0.        , ...,  574.46932602,
           0.        ,    0.        ],
       [   0.        ,    0.        ,    0.        , ...,    0.        ,
         670.41536276,    0.        ],
       [   0.        ,    0.        ,    0.        , ...,    0.        ,
           0.        , 1544.10679346]])
    """
    
"""
Making Predictions from the Decomposed Matrices

I now have everything I need to make movie ratings predictions for every user. 
I can do it all at once by following the math and matrix multiply 
U, Σ, and VT
 back to get the rank k=40 approximation of ratings_new_df_matrix
 
 I also need to add the user means back to get the actual star ratings prediction.
"""

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

all_user_predicted_ratings

"""
aftre getting the approximation matrix  for all user ,
we will be building a function  to recommend movies for any user 

This can  be done , by returning  the movies with the highest predicted rating 
that the specified user hasn't already rated
"""

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = ratings_new_df.columns)
preds_df.head(2)
"""
 here the records rows are user id 
 movie id are the columns
 
MovieID      1         2         3         4         5         6         7     \
0        3.814181  0.294102 -0.155784 -0.028457  0.061794 -0.144988 -0.111370   
1        1.089991  0.503913  0.204283  0.030002  0.017701  0.968582  0.018518   

MovieID      8         9         10      ...         3943      3944      3945  \
0        0.176379 -0.046708 -0.217093    ...    -0.015179  0.019517  0.041431   
1        0.062975  0.175583  1.481338    ...    -0.039839 -0.013562 -0.009879   

MovieID      3946      3947      3948      3949      3950      3951      3952  
0       -0.017069 -0.089792  0.277882  0.011418  0.013654  0.015451  0.062495  
1        0.046846 -0.018421  0.067193 -0.372172 -0.083298 -0.046230 -0.158692
"""
preds_df.iloc[0]## shows is the ratings of all the movies for user  having user id 0 value  


def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) #sort the user pred values descending wise
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, ]
                      )### select first num_recoemmendations records , till  last column 

    return user_full, recommendations

########################################################
"""user_row_number=15-1

sorted_user_predictions = preds_df.iloc[14].sort_values(ascending=False)
sorted_user_predictions.size

user_data = ratings_df[ratings_df.UserID == (14)]

user_data##25

user_full= (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )
movies_df.count()###3883

user_full.count()##25

recommendations= (movies_df[movies_df['MovieID'].isin(user_full['MovieID'])])
         
    recommendations.dtypes                  

recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

"""


### call the function 

"""
we are trying to find out the top 10 recommendations for user having userid :837 
"""
already_rated, predictions = recommend_movies(preds_df, 837, movies_df, ratings_df, 10)

"""
User 837 has already rated 69 movies.
Recommending highest 10 predicted ratings movies not already rated.
"""
already_rated.head(10)

"""
UserID  MovieID  Rating  Timestamp  \
36     837      858       5  975360036   
35     837     1387       5  975360036   
65     837     2028       5  975360089   
63     837     1221       5  975360036   
11     837      913       5  975359921   
20     837     3417       5  975360893   
34     837     2186       4  975359955   
55     837     2791       4  975360893   
31     837     1188       4  975360920   
28     837     1304       4  975360058   

                                        Title                   Genres  
36                      Godfather, The (1972)       Action|Crime|Drama  
35                                Jaws (1975)            Action|Horror  
65                 Saving Private Ryan (1998)         Action|Drama|War  
63             Godfather: Part II, The (1974)       Action|Crime|Drama  
11                 Maltese Falcon, The (1941)        Film-Noir|Mystery  
20                 Crimson Pirate, The (1952)  Adventure|Comedy|Sci-Fi  
34                Strangers on a Train (1951)       Film-Noir|Thriller  
55                           Airplane! (1980)                   Comedy  
31                   Strictly Ballroom (1992)           Comedy|Romance  
28  Butch Cassidy and the Sundance Kid (1969)    Action|Comedy|Western  
"""

predictions.dtypes
"""
MovieID     int64
Title      object
Genres     object
"""
"""
MovieID                             Title                         Genres
596       608                      Fargo (1996)           Crime|Drama|Thriller
1848     1953     French Connection, The (1971)    Action|Crime|Drama|Thriller
581       593  Silence of the Lambs, The (1991)                 Drama|Thriller
516       527           Schindler's List (1993)                      Drama|War
2085     2194          Untouchables, The (1987)             Action|Crime|Drama
1175     1214                      Alien (1979)  Action|Horror|Sci-Fi|Thriller
1235     1284             Big Sleep, The (1946)              Film-Noir|Mystery
1849     1954                      Rocky (1976)                   Action|Drama
106       111                Taxi Driver (1976)                 Drama|Thriller
1196     1240            Terminator, The (1984)         Action|Sci-Fi|Thriller
"""

