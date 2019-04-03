# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:43:18 2019

@author: USER

Purpose:
    The dataset we’ll be using is a subset of the last.fm dataset. 
    It’s a sparse matrix containing 285 artists and 1226 users and contains what users have listened 
    to what artists.
    We don’t care about when the user listened to the song as we assume
    that music tastes are fairly static over time.
    
    data set used :LastFmMatrix 
    The dataset we’ll be using is a subset of the last.fm dataset.
    It’s a sparse matrix containing 285 artists and 1226 users and 
    contains what users have listened to what artists. We don’t care about when
    the user listened to the song as we assume that music tastes are
    fairly static over time.
"""

import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

##C:\Users\USER\Documents\datasets\movie

Lastfm_df=pd.read_csv("C:/Users/USER/Documents/datasets/lastfm.csv",encoding="latin-1")

"""
Item-Item calculations
As mentioned above we start with computing the item-item relationships of our songs. 
Our final goal here is to construct a new item by item matrix containing the weights (relationships) 
between each of our songs where a perfect correlation equals 1 and no correlation at all equals 0

To get this relationship between items we calculate the cosine similarity
Basically what it means is 
that we take the dot-product of our different item-vectors and divide 
it by the product of the normalized vectors.

Item-item summary:
Normalize user vectors to unit vectors.
Construct a new item by item matrix.
Compute the cosine similarity between all items in the matrix.
"""
Lastfm_df.head()
# Create a new dataframe without the user ids.
data = Lastfm_df.drop('user', 1)


"""
now first we normalize the user ratings in data_items
numpy array axiz =1 means for each user , that is calculation alomg rows
first find out the sequare root of sum of squares 
"""
magnitude = np.sqrt(np.square(data).sum(axis=1))

data = data.divide(magnitude, axis='index')

"""
run calculate_similarity to generate a new item by item matrix with our 
similarities called data_matrix
"""

def calculate_similarity(data):
    """Calculate the column-wise cosine similarity for a sparse
    matrix. Return a new dataframe matrix with similarities.
    """
    data_sparse = sparse.csr_matrix(data)
    similarities = cosine_similarity(data_sparse.transpose())
    sim = pd.DataFrame(data=similarities, index= data.columns, columns= data.columns)
    return sim

# Build the similarity matrix
data_matrix = calculate_similarity(data)

data_matrix.head(3)

"""
  a perfect circle      abba     ac/dc  adam green  aerosmith  \
a perfect circle           1.00000  0.000000  0.009250    0.032188   0.066869   
abba                       0.00000  1.000000  0.024286    0.009154   0.029176   
ac/dc                      0.00925  0.024286  1.000000    0.072087   0.148919   

                       afi       air  alanis morissette  alexisonfire  \
a perfect circle  0.000000  0.038886           0.039923           0.0   
abba              0.000000  0.005186           0.026254           0.0   
ac/dc             0.058515  0.057011           0.022184           0.0   

                  alicia keys      ...       timbaland  tom waits      tool  \
a perfect circle     0.000000      ...        0.029562   0.057362  0.349047   
abba                 0.000000      ...        0.000000   0.000000  0.000000   
ac/dc                0.055834      ...        0.019406   0.039212  0.031971   

                  tori amos    travis   trivium        u2  underoath  \
a perfect circle   0.106208  0.018213  0.079469  0.017868   0.067862   
abba               0.020101  0.014697  0.000000  0.055088   0.000000   
ac/dc              0.017305  0.000000  0.076235  0.084368   0.008948   

                   volbeat  yann tiersen  
a perfect circle  0.044661           0.0  
abba              0.010399           0.0  
ac/dc             0.081664           0.0

"""

"""

Lastly we print , 20 most similar  items pertaining to a particular artist 
like alanis morissette
"""

# Lets get the top 11 similar artists for alanis morissette
df_sorted = data_matrix.sort_values(by='alanis morissette', ascending=False).head(20)
  ## or we can try the below both yield the sam eresults 
df_sorted_final = data_matrix.loc['alanis morissette'].nlargest(20)
print(df_sorted_final)


"""
alanis morissette        1.000000
tori amos                0.240970
pearl jam                0.196749
kelly clarkson           0.172014
dido                     0.171421
james blunt              0.159333
jack johnson             0.138514
red hot chili peppers    0.133195
avril lavigne            0.128762
alicia keys              0.116254
bruce springsteen        0.110765
coldplay                 0.108603
nirvana                  0.097627
damien rice              0.095894
eric clapton             0.092657
christina aguilera       0.091617
norah jones              0.090808
tegan and sara           0.086101
clueso                   0.083248
motorhead                0.082461
"""
