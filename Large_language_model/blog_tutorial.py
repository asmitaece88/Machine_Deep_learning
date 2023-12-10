###added asmita -28-11
#OPENAI_API_KEY=""
#https://betterprogramming.pub/vector-search-using-openai-embeddings-with-weaviate-ebb11f2c18cf
# Vector Search Using OpenAI Embeddings With Weaviate
# Perform vector search on text data using a vector database    


import openai
import pandas as pd
import weaviate
from openai.embeddings_utils import get_embedding
import os 


openai.api_key = ""
os.environ['OPENAI_API_KEY']=openai.api_key
# client = weaviate.Client(url="http://localhost:8080")

# Instantiate the client
auth_config = weaviate.AuthApiKey(api_key="")
client = weaviate.Client(
    url="", # Replace w/ your Weaviate cluster URL
    auth_client_secret=auth_config,
    additional_headers={
        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY'] # Replace with your OpenAI key
        }
)

#This function simply reads the JSON file 
# #and converts it into a pandas data frame. There is only one column called “text”.
def read_json_file():
    filename = "C:\\Users\\09029O744\\Documents\\GitHub\\main_3\\jup_files\\archive\\vector_search_using_openaiembd_weaivate\\history_text.json"
    df = pd.read_json(filename)
    return df

# Working with OpenAI embeddings
# To do a vector search across our text data we first need to convert our text 
# into a vector-based representation.
# This is where OpenAI’s embedding API comes in handy.
# We will create a new column in our data frame called “embedding” 
# that will contain the vector representation of the text in that row.
def generate_data_embeddings(df):
    df['embedding']=df['text'].apply(lambda row :get_embedding(row ,engine="text-embedding-ada-002"))
    return df


# : Create the Weaviate schema
# Weaviate stores data as JSON objects inside of a collection 
# which they call “classes”. Weaviate can 
# not only store the JSON object itself, but also the vector-based 
# epresentation of that JSON object.



# We need to tell Weaviate how our data looks by providing a schema.
# This vector database has many features and can vectorize data automatically. 
# However, we want to provide our own OpenAI embeddings. In this case, it’s 
# important that we set “vectorizer” to “none”, otherwise Weaviate will use one 
# f its vectorizers to create the embedding.

def weaviate_create_schema():
    schema = {
        "classes": [{
            "class": "HistoryText",
            "description": "Contains the paragraphs of text along with their embeddings",
            "vectorizer": "none",
            "properties": [{
                "name": "content",
                "dataType": ["text"],
            }]
        }]
    }
    client.schema.create(schema)
    
    
## function to delete the schema 
def weaviate_delete_schema():
    client.schema.delete_class("HistoryText")
    
#Adding data to our vector database
# The Weaviate python client provides various ways to add data to the database. 
# As of the time of writing this article, 
# one of the newer ways to add data is by using automated batches.
# Weaviate will automatically create our data objects when the specified batch size is met.   


def weaviate_add_data(df):
    client.batch.configure(batch_size=10)
    with client.batch as batch:
        for index, row in df.iterrows():
            text = row['text']
            ebd = row['embedding']
            batch_data = {
                "content": text
            }
            batch.add_data_object(data_object=batch_data, class_name="HistoryText", vector=ebd)

    print("Data Added!")
    
    
# Querying the data
# Our objective was to find texts similar to our input text. 
# The text in our vector database is stored as embeddings. 
# In order to do a vector search, our input text needs to be converted to an embedding as well.
# The second parameter to this function “k” is the number of objects we want to return that 
# is closest to our input.

# We want to find the closest vectors to our input vector
# so we chain the .with_near_vector() function. Inside the get function,
# we are requesting both “content” which is our raw text along with “_additional {certainty}”.
# The “certainty” is essentially telling us how close the returned vector is to our input vector.

def query(input_text, k):
    input_embedding = get_embedding(input_text, engine="text-embedding-ada-002")
    vec = {"vector": input_embedding}
    result = client \
        .query.get("HistoryText", ["content", "_additional {certainty}"]) \
        .with_near_vector(vec) \
        .with_limit(k) \
        .do()

    output = []
    closest_paragraphs = result.get('data').get('Get').get('HistoryText')
    for p in closest_paragraphs:
        output.append(p.get('content'))

    return output

if __name__ == "__main__":
    #weaviate_delete_schema
    dataframe = read_json_file()
    dataframe = generate_data_embeddings(dataframe)
    weaviate_create_schema()
    weaviate_add_data(dataframe)
    # ONLY RUN THE ABOVE 4 LINES ONCE

    input_text = "Fertile land"
    k_vectors = 3
    
    result = query(input_text, k_vectors)
    for text in result:
        print(text)         