import os
from dotenv import load_dotenv

from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")


llm_name = "gpt-3.5-turbo"
client = OpenAI(api_key=openai_key)

response = client.chat.completions.create(
    model=llm_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "who is Nelson Mandela?"},
    ],
)
#print(response.choices[0].message.content)

class Agent:
    def __init__(self,system=""):
        self.system=system
        self.messages=[]        
        if system:
            self.messages.append(({"role":"system","content":system}))
    
    def __call__(self,message):
        self.messages.append({"role":"user","content":message})
        result =self.execute()  
        self.messages.append({"role":"assistant","content":result})
        return result

    def execute(self) :
        response = client.chat.completions.create(model=llm_name ,
                                                  temperature=0,
                                                  messages=self.messages)
        
        return response.choices[0].message.content

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

planet_mass:
e.g. planet_mass: Earth
returns the mass of a planet in the solar system

Example session:

Question: What is the combined mass of Earth and Mars?
Thought: I should find the mass of each planet using planet_mass.
Action: planet_mass: Earth
PAUSE

You will be called again with this:

Observation: Earth has a mass of 5.972 * 10^24 kg

You then output:

Answer: Earth has a mass of 5.972 * 10^24 kg

Next, call the agent again with:

Action: planet_mass: Mars
PAUSE

Observation: Mars has a mass of 0.64171 * 10^24 kg

You then output:

Answer: Mars has a mass of 0.64171 * 10^24 kg

Finally, calculate the combined mass.

Action: calculate: 5.972 + 0.64171
PAUSE

Observation: The combined mass is 6.61371 * 10^24 kg

Answer: The combined mass of Earth and Mars is 6.61371 * 10^24 kg
""".strip()        


# Implement the functions actions
def calculate(what):
    return eval(what)


def planet_mass(name):
    masses = {
        "Mercury": 0.33011,
        "Venus": 4.8675,
        "Earth": 5.972,
        "Mars": 0.64171,
        "Jupiter": 1898.19,
        "Saturn": 568.34,
        "Uranus": 86.813,
        "Neptune": 102.413,
    }
    return f"{name} has a mass of {masses[name]} × 10^24 kg"

known_actions = {"calculate": calculate, "planet_mass": planet_mass}



### next using a loop tp automate our agent 

# ----- Final solution - Automate our AI Agent ------
# Create a loop to automate the agent until the agent returns an answer

import re

action_re = re.compile(r"^Action: (\w+): (.*)$")
## Create a query function
def query(question, max_turns=10):
    i = 0
    bot = Agent(prompt)
    print(f"##debug ##printing the bot:{bot}  ")
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(f"##debug ##printing the result:{result}  ")
        print(result)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        print(f"##debug ##printing the actions:{actions}  ")
        if actions:
            # There is an action to run
            print(f"printing the actions[0].groups() :{actions[0].groups()}")
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(f"#printing action, action_input:{action}, \n:{action_input}  ")
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
        
# New Scenario: Calculating Combined Mass of Earth and Jupiter
question = "What is the combined mass of Earth and Jupiter and Saturn and Venus?"
query(question)        

### result output 

##debug ##printing the bot:<__main__.Agent object at 0x000002DAF6DE6DD0>  
##debug ##printing the result:Thought: I need to find the mass of each planet - Earth, Jupiter, Saturn, and Venus - using the planet_mass action, and then calculate the combined mass.

# Action: planet_mass: Earth
# PAUSE  
# Thought: I need to find the mass of each planet - Earth, Jupiter, Saturn, and Venus - using the planet_mass action, and then calculate the combined mass.

# Action: planet_mass: Earth
# PAUSE
# ##debug ##printing the actions:[<re.Match object; span=(0, 26), match='Action: planet_mass: Earth'>]
# printing the actions[0].groups() :('planet_mass', 'Earth')
# #printing action, action_input:planet_mass,
# :Earth
#  -- running planet_mass Earth
# Observation: Earth has a mass of 5.972 × 10^24 kg
# ##debug ##printing the result:Answer: Earth has a mass of 5.972 × 10^24 kg

# Action: planet_mass: Jupiter
# PAUSE
# Answer: Earth has a mass of 5.972 × 10^24 kg

# Action: planet_mass: Jupiter
# PAUSE
# ##debug ##printing the actions:[<re.Match object; span=(0, 28), match='Action: planet_mass: Jupiter'>]
# printing the actions[0].groups() :('planet_mass', 'Jupiter')
# #printing action, action_input:planet_mass,
# :Jupiter
#  -- running planet_mass Jupiter
# Observation: Jupiter has a mass of 1898.19 × 10^24 kg
# ##debug ##printing the result:Answer: Jupiter has a mass of 1898.19 × 10^24 kg

# Action: planet_mass: Saturn
# PAUSE
# Answer: Jupiter has a mass of 1898.19 × 10^24 kg

# Action: planet_mass: Saturn
# PAUSE
# ##debug ##printing the actions:[<re.Match object; span=(0, 27), match='Action: planet_mass: Saturn'>]
# printing the actions[0].groups() :('planet_mass', 'Saturn')
# #printing action, action_input:planet_mass,
# :Saturn
#  -- running planet_mass Saturn
# Observation: Saturn has a mass of 568.34 × 10^24 kg
# ##debug ##printing the result:Answer: Saturn has a mass of 568.34 × 10^24 kg

# Action: planet_mass: Venus
# PAUSE
# Answer: Saturn has a mass of 568.34 × 10^24 kg

# Action: planet_mass: Venus
# PAUSE
# ##debug ##printing the actions:[<re.Match object; span=(0, 26), match='Action: planet_mass: Venus'>]
# printing the actions[0].groups() :('planet_mass', 'Venus')
# #printing action, action_input:planet_mass,
# :Venus
#  -- running planet_mass Venus
# Observation: Venus has a mass of 4.8675 × 10^24 kg
# ##debug ##printing the result:Answer: Venus has a mass of 4.8675 × 10^24 kg

# Action: calculate: 5.972 + 1898.19 + 568.34 + 4.8675
# PAUSE
# Answer: Venus has a mass of 4.8675 × 10^24 kg

# Action: calculate: 5.972 + 1898.19 + 568.34 + 4.8675
# PAUSE
# ##debug ##printing the actions:[<re.Match object; span=(0, 52), match='Action: calculate: 5.972 + 1898.19 + 568.34 + 4.8>]
# printing the actions[0].groups() :('calculate', '5.972 + 1898.19 + 568.34 + 4.8675')
# #printing action, action_input:calculate,
# :5.972 + 1898.19 + 568.34 + 4.8675
#  -- running calculate 5.972 + 1898.19 + 568.34 + 4.8675
# Observation: 2477.3695
# ##debug ##printing the result:Answer: The combined mass of Earth, Jupiter, Saturn, and Venus is 2477.3695 × 10^24 kg  
# Answer: The combined mass of Earth, Jupiter, Saturn, and Venus is 2477.3695 × 10^24 kg
# ##debug ##printing the actions:[]