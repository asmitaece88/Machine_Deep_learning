# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:03:01 2019

@author: asmita chatteree

 data source : kaggle.com
 
 methodology applied : content based filtering 
 
 
 Purpose: find out the recoemmendations of other hotels , if you select a partocular hotel
 based on the description provided in a marketing website 
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import random
###import plotly.graph_objs as go
###import plotly.plotly as py
from IPython.core.interactiveshell import InteractiveShell

hotel_df = pd.read_csv( "C:/Users/USER/Documents/datasets/Seattle_Hotels.csv",encoding="latin-1" )
hotel_df.head(1)

hotel_df['name'].nunique()
##152 

print('Unique number of hotels', hotel_df['name'].nunique())
"""
Unique number of hotels 152
"""

### checking the column 
hotel_df.dtypes
"""
name       object
address    object
desc       object
dtype: object
"""

## now write a function  which prints the description and the hotel name of the index passes 
def hotel_description(index):
    example = hotel_df[hotel_df.index == index][['desc', 'name']].values[0]
    print(example[0])
    print('Name:', example[1])
    
    
    """
    check desciption  and the hotel name of the index 
    """
    hotel_description(2)
    """
    Located in the heart of downtown Seattle, the award-winning 
Crowne Plaza Hotel Seattle ? Downtown offers an exceptional blend of service, style and comfort. 
You?ll notice Cool, Comfortable and Unconventional touches that set us apart as soon as you step inside.
 Marvel at stunning views of the city lights while relaxing in our new Sleep Advantage? 
 Beds. Enjoy complimentary wireless 
 Internet throughout the hotel and amenities to help you relax like our Temple Spa?
 Sleep Tight Amenity kits featuring lavender spray and lotions to help you rejuvenate and unwind. 
 Enjoy an invigorating workout at our 24-hour fitness center, get dining suggestions from our expert concierge or savor sumptuous cuisine
 at our Regatta Bar & Grille restaurant where you can enjoy Happy Hour in our lounge daily from 4pm - 7pm and monthly drink specials. 
 Come and experience all that The Emerald City has to offer with us!
Name: Crowne Plaza Seattle Downtown
"""
        
hotel_df['desc']

"""
Finding out the total word count of all the descriptions of all the hotels 
""""
hotel_df['word_count'] = hotel_df['desc'].apply(lambda x: len(str(x).split()))
        
desc_lengths = list(hotel_df['word_count'])

"""
finding  out the length of all the descriptions ,
average length of description ,
minimum of description length 
maximum of description length
"""

print("Number of descriptions:",len(desc_lengths),
      "\nAverage word count", np.average(desc_lengths),
      "\nMinimum word count", min(desc_lengths),
      "\nMaximum word count", max(desc_lengths))


"""
Number of descriptions: 152 
Average word count 156.94736842105263 
Minimum word count 16 
Maximum word count 494
"""

"""
Preprocessing hotel description text
"""

ENGLISH_STOPWORDS = set(stopwords.words('english'))


## define a function , which will  remove xml tags , transforms all texts to lowercase 
## special characaters  , removes stopwords 


def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove xml tags
    text=re.sub("</?.*?>"," <> ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    text = ' '.join(word for word in text.split() if word not in ENGLISH_STOPWORDS) # remove stopwors from text
   
    return text


hotel_df['desc_clean'] = hotel_df['desc'].apply(pre_process)


###test example  to check whether the function strips unwanted characters from a text 
test_example_1 = hotel_df[hotel_df.index==0][['desc_clean', 'name']].values[0]
test_example_1[0]
"""
'located southern tip lake union hilton garden inn seattle downtown hotel perfectly
 located business leisure neighborhood home numerous major international companies
 including amazon google bill melinda gates foundation wealth eclectic restaurants
 bars make area seattle one sought locals visitors proximity lake union allows visitors take pacific northwest majestic scenery enjoy outdoor activities 
 like kayaking sailing sq ft versatile space complimentary business center state art v technology helpful staff guarantee conference cocktail reception
 wedding success refresh sparkling saltwater pool energize latest equipment hour fitness center tastefully decorated flooded natural light guest rooms suites 
 offer everything need relax stay productive unwind bar enjoy american cuisine breakfast lunch dinner restaurant hour pavilion pantry stocks variety snacks drinks sundries'
"""
test_example_1[1]
"""
'Hilton Garden Seattle Downtown'
"""


def print_description(index):
    example = hotel_df[hotel_df.index == index].[['desc_clean', 'name']].values[0]
    print(example[0])
    print("****************")
    print('Name:', example[1])
        
        
 ## checking the description and name of the 11th hotel        
print_description(10)
"""
soak vibrant scene living room bar get mix live music dj series heading memorable dinner trace offering inspired seasonal 
fare award winning atmosphere missed culinary experience downtown seattle work next morning fit state art fitness center wandering explore many area nearby attractions 
including pike place market pioneer 
square seattle art museum always got covered time w seattle signature whatever whenever service wish truly command
****************
Name: W Seattle
"""

hotel_df.set_index('name', inplace = True)

hotel_df.head(3)



tf_val = TfidfVectorizer(analyzer='word', min_df=0, stop_words='english')
tfidf_matrix = tf_val.fit_transform(hotel_df['desc_clean'])

"""
Now find out the similarity between the Fodf values of various words in the vocabulary
"""

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

"""
array([[1.        , 0.05178023, 0.09116494, ..., 0.05326527, 0.01055984,
        0.02426098],
       [0.05178023, 1.        , 0.08165308, ..., 0.05921971, 0.01474194,
        0.03336386],
       [0.09116494, 0.08165308, 1.        , ..., 0.09361202, 0.02487061,
        0.03767873],
       ...,
       [0.05326527, 0.05921971, 0.09361202, ..., 1.        , 0.05047708,
        0.03126965],
       [0.01055984, 0.01474194, 0.02487061, ..., 0.05047708, 1.        ,
        0.00641147],
       [0.02426098, 0.03336386, 0.03767873, ..., 0.03126965, 0.00641147,
        1.        ]])
"""


indices = pd.Series(hotel_df.index)

##chck the 6th indexed hotel name 
indices[5]

##chck  10 indices 

indices[:10]

"""
0    Hilton Garden Seattle Downtown
1            Sheraton Grand Seattle
2     Crowne Plaza Seattle Downtown
3     Kimpton Hotel Monaco Seattle 
4                The Westin Seattle
5       The Paramount Hotel Seattle
6                    Hilton Seattle
7                     Motif Seattle
8                   Warwick Seattle
9        Four Seasons Hotel Seattle
"""

indices[indices == "The Paramount Hotel Seattle"].index[0]
"""
Out[245]: 5
"""

indices[indices == "citizenM Seattle South Lake Union hotel"].index[0]
"""151
"""
##  below code finds out the similarity of the 6th indexed hotel with all the others in the corpus 
"""score_series_test = pd.Series(cosine_similarities[5]).sort_values(ascending = False)
top_10_indexes_test = list(score_series_test.iloc[1:11].index)
recommended_hotels= []
 for i in top_10_indexes_test:
        recommended_hotels.append(list(hotel_df.reset_index().name)[i])
        
        
 recommended_hotels      
""" 

"""
now mention a function  to provide recommemdations 
"""

def recommendations(name, cosine_similarities = cosine_similarities):
    
    recommended_hotels = []
    
    # gettin the index of the hotel that matches the name
    idx = indices[indices == name].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar hotels except itself
    #### fetch top 10 rows 
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the names of the top 10 matching hotels
    for i in top_10_indexes:
        recommended_hotels.append(list(hotel_df.reset_index().name)[i])
        
    return recommended_hotels



"""
find out the recommendations, if you select a particular hotel  
say Mozart Guest House

"""

recommendations("Mozart Guest House")
"""
['City Hostel Seattle',
 'Hotel Seattle',
 'Crown Inn Motel',
 'Motel 6 Seattle Sea-Tac Airport South',
 '11th Avenue Inn Bed and Breakfast',
 "Mildred's Bed and Breakfast",
 'Hotel Hotel',
 'Ramada by Wyndham SeaTac Airport',
 'Country Inn & Suites by Radisson, Seattle-Tacoma International Airport',
 'Days Inn Seattle South Tukwila']
"""


"""
11th Avenue Inn Bed and Breakfast
"""

recommendations("11th Avenue Inn Bed and Breakfast")

"""
['Hotel Seattle',
 'Inn at the Market',
 'Inn at Queen Anne',
 'The Bacon Mansion Bed and Breakfast',
 'Shafer Baillie Mansion Bed & Breakfast',
 'Holiday Inn Seattle Downtown',
 'Silver Cloud Hotel - Seattle Broadway',
 'Gaslight Inn',
 'Bed and Breakfast Inn Seattle',
 'Holiday Inn Express & Suites Seattle-City Center']

"""



"""
hence we use TFIDF to first calcuate the TFidf scores , and then fid out the cosine similarity
of each pair of hotels based on the descriptio given in th marketing website 

Based on the similarity , we  recoemmend a set of hotels for a particular hotel 

This is all about content based similaity , in which we create  a  item ( hotel)
based profile for each hotel 



"""
