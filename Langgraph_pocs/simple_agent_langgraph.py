import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")


llm_name = "gpt-3.5-turbo"

client = OpenAI(api_key=openai_key)
model = ChatOpenAI(api_key=openai_key, model=llm_name)

# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages] 
    
    
    
def bot_func(state: State):
    # print(state.items())
    print(f"printing --1---:{state}")
    print(f"---2---")
    print(state["messages"])
    print(f"---3---")
    return {"messages": [model.invoke(state["messages"])]}    


graph_builder=StateGraph(State)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot",bot_func)


# STEP 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")


# STEP 4: and end point to the graph
graph_builder.set_finish_point("bot")


# STEP 5: Compile the graph
graph = graph_builder.compile()


res = graph.invoke({"messages": ["Hello, how are you?"]})
res_messages=res["messages"]
print(f"finally printing :{res_messages}")


# --result output 

# printing --1---:{'messages': [HumanMessage(content='Hello, how are you?', id='05976be1-2ac6-40b5-a4e4-1bf92793a826')]}
# ---2---
# [HumanMessage(content='Hello, how are you?', id='05976be1-2ac6-40b5-a4e4-1bf92793a826')]
# ---3---
# finally printing :[HumanMessage(content='Hello, how are you?', id='05976be1-2ac6-40b5-a4e4-1bf92793a826'), AIMessage(content="Hello! I'm just a virtual assistant, so I don't have feelings, but I'm here to help you with any questions or tasks you may have. How can I assist you today?", response_metadata={'token_usage': {'completion_tokens': 39, 'prompt_tokens': 13, 'total_tokens': 52}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-e8d02539-7742-42db-b9d3-f71360b2b109-0', usage_metadata={'input_tokens': 13, 'output_tokens': 39, 'total_tokens': 52})]