# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:42:16 2019

@author: USER
"""


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
###nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


###check the stopwords 

en_stops = set(stopwords.words('english'))

### 2 documents are 2 sentences 
docA = "The car is driven on the road"
docB = "The truck is driven on the highway"

word_tokens_A = word_tokenize(docA) 
word_tokens_B = word_tokenize(docB) 

filtered_doca = [w for w in word_tokens_A if not w in stopwords.words()]
filtered_docb = [w for w in word_tokens_B if not w in stopwords.words()]



 """
  find out the filtered detokenized sentences 
"""      
doca_1 = TreebankWordDetokenizer().detokenize(filtered_doca)


docb_1 = TreebankWordDetokenizer().detokenize(filtered_docb)

"""
call the tfid vectorizer
"""
tfidf = TfidfVectorizer()

tfidf

response = tfidf.fit_transform([docA, docB])

print(response)



"""
get the features of  the 2 document sets 
"""
feature_names = tfidf.get_feature_names()

for col in response.nonzero()[1]:
    print (feature_names[col], ' - ', response[0, col])
    
    """
    leaving the common words , the word "car" and "road"
    hve the highest Tf-IDf that is 0.42471718586982765
    """