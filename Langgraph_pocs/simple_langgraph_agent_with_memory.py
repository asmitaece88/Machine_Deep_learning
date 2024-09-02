import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")


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


graph_builder = StateGraph(State)

# create tools
tool = TavilySearchResults(max_results=2)
tools = [tool]
# rest = tool.invoke("What is the capital of France?")
# print(rest)

model_with_tools = model.bind_tools(tools)

# Below, implement a BasicToolNode that checks the most recent
# message in the state and calls tools if the message contains tool_calls
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition


def bot(state: State):
    # print(state.items())
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}


# instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)  # Add the node to the graph


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)


# STEP 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")

# ADD MEMORY NODE
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver


# class MySqliteSaver(BaseCheckpointSaver):
#     pass

##memory = SqliteSaver.from_conn_string("memory")

# ... define the graph
memory = MemorySaver()

# STEP 5: Compile the graph
graph = graph_builder.compile(checkpointer=memory)
# MEMORY CODE CONTINUES ===
# Now we can run the chatbot and see how it behaves
# PICK A TRHEAD FIRST
config = {
    "configurable": {"thread_id": 1}
}  # a thread where the agent will dump its memory to
user_input = "Hi there! My name is Bond. and I have been happy for 100 years"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()
    
    
user_input = "do you remember my name, and how long have I been happy for?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()
    
    
## result output 
================================ Human Message =================================

Hi there! My name is Bond. and I have been happy for 100 years
[HumanMessage(content='Hi there! My name is Bond. and I have been happy for 100 years', id='0433dee8-87c4-421b-aa14-9193ef5bd044')]
================================== Ai Message ==================================

Hello Bond! It's great to hear that you've been happy for 100 years. That's quite an achievement! How can I assist you today? 
================================ Human Message =================================

do you remember my name, and how long have I been happy for?
[HumanMessage(content='Hi there! My name is Bond. and I have been happy for 100 years', id='0433dee8-87c4-421b-aa14-9193ef5bd044'), AIMessage(content="Hello Bond! It's great to hear that you've been happy for 100 years. That's quite an achievement! How can I assist you today?", response_metadata={'token_usage': {'completion_tokens': 32, 'prompt_tokens': 98, 'total_tokens': 130}, 'model_name': 'gp0}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-1951bb91-0636-4893-be65-e119e1cfbe32-0', usage_metadata={'input_tokens': 98, 'output_tokens': 32, 'total_tokens': 130}), HumanMessage(content='do you remember my name, and how long have I been happy for?', id='30d2dec5-43f5-4eb2-994b-0287368aada3')]
================================== Ai Message ==================================

Hello Bond! Yes, I remember your name. You mentioned that you have been happy for 100 years. If there's anything specific you'd like to know or discuss, feel free to let me know!
    

    