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
# print(rest[0]['content'],rest[1]['content'])

model_with_tools=model.bind_tools(tools)
# res=model_with_tools.invoke("what is the capital of India?")
# print(res)

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
graph_builder.add_node("tools", tool_node)  # Add the tools node to the graph. tools is the list of all tool calls

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,## this is the path 
)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)


# STEP 3: Add an entry point to the graph
graph_builder.set_entry_point("bot")

# STEP 5: Compile the graph
graph = graph_builder.compile()

from langchain_core.messages import BaseMessage

while True:
    user_input=input("User:")
    if user_input.lower() in ["quit","exit","q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            if isinstance(value["messages"][-1],BaseMessage):
                print("Assistant:",value["messages"][-1].content)

""" result , 4 quesios were entered  for which the response is as below :
->what is the capital of france ?
->who was the first prime minister?
->who was the first prime minister of france ?

User:what is the capital of france ?
[HumanMessage(content='what is the capital of france ?', id='3af5d046-7d02-4fb2-b99c-4000db0469a1')]
Assistant:
Assistant: [{"url": "https://en.wikipedia.org/wiki/List_of_capitals_of_France", "content": "Learn about the historical and current capitals of France, from Tournai to Paris. See the chronology of changes, the reasons for relocations, and the provisional seats of the government."}, {"url": "https://www.britannica.com/facts/Paris", "content": "Learn about Paris, the capital of France, and its history, culture, landmarks, and people. Find out why Paris is called \"the City of Light\" and when it was founded, liberated, and attacked."}]
User:who was the first prime minister?
[HumanMessage(content='who was the first prime minister?', id='81d6f275-220c-4e0b-acb8-d274181626a9')]
Assistant:
Assistant: [{"url": "https://en.wikipedia.org/wiki/List_of_prime_ministers_of_the_United_Kingdom", "content": "The first to use the title in an official act was Benjamin Disraeli, who, in 1878, signed the Treaty of Berlin as \"Prime Minister of Her Britannic Majesty\".[8]\nIn 1905, the post of prime minister was officially given recognition in the order of precedence,[9] with the incumbent Henry Campbell-Bannerman the first officially referred to as \"prime minister\".\n The term was regularly, if informally, used of Robert Walpole by the 1730s.[2] It was used in the House of Commons as early as 1805,[3] and it was certainly in parliamentary use by the 1880s.[4]\nModern historians generally consider Robert Walpole, who led the government of the Kingdom of Great Britain for over twenty years from 1721,[5] as the first prime minister. Contents\nList of prime ministers of the United Kingdom\nThe prime minister of the United Kingdom is the principal minister of the crown of His Majesty's Government, and the head of the British Cabinet. The last lords high treasurer, Sidney Godolphin, 1st Earl of Godolphin (1702\u20131710) and Robert Harley, 1st Earl of Oxford (1711\u20131714),[17] ran the government of Queen Anne.[18]\nFrom 1707 to 1721[edit]\nFollowing the succession of George\u00a0I in 1714, the arrangement of a commission of lords of the Treasury (as opposed to a single lord high treasurer) became permanent.[19] The first prime minister of the current United Kingdom of Great Britain and Northern Ireland upon its effective creation in 1922 (when 26 Irish counties seceded and created the Irish Free State) was Bonar Law,[10] although the country was not renamed officially until 1927, when Stanley Baldwin was the serving prime minister.[11] The incumbent prime minister is Rishi Sunak.\n"}, {"url": "https://www.britannica.com/place/list-of-prime-ministers-of-Great-Britain-and-the-United-Kingdom-1800350", "content": "Find out who was the first prime minister of Great Britain and the United Kingdom, and see the chronological list of all the holders of this office since 1721. Learn about the history and role of the prime minister in Britannica's articles."}]
User:who was the first prime minister of france ?
[HumanMessage(content='who was the first prime minister of france ?', id='e8d48325-212c-4f66-bec5-b6d3205f7f6d')]
Assistant: 
Assistant: [{"url": "https://www.firstpost.com/world/french-president-macron-to-meet-with-former-socialist-party-leader-ahead-of-prime-minister-announcement-13810584.html", "content": "France's next prime minister will have the daunting task of trying to drive reforms and the 2025 budget through a hung parliament, as France is under pressure from the European Commission and bond markets to reduce its deficit. ... The left's New Popular Front alliance came first but Macron ruled out asking it to form a government after ..."}, {"url": "https://www.politico.eu/article/emmanuel-macron-france-president-politics-rules-government/", "content": "He backed Cazeneuve, a seasoned politician who was Hollande's prime minister and represents the French left's most centrist-leaning fringe, he added. Cazeneuve left the Socialist party in 2022 to protest the alliance with Jean-Luc M\u00e9lenchon's France Unbowed."}]
User:exit
Goodbye!
"""