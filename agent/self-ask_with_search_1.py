from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.llms import Fireworks
from langchain_community.tools.tavily_search import TavilyAnswer
from yucl.utils import ChatOpenAI

tools = [TavilyAnswer(max_results=1, name="Intermediate Answer")]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/self-ask-with-search")

import os

os.environ["FIREWORKS_API_KEY"] = "wrTGJ0yeD9BTheuMyeLl6kqic2rG3GtoL3jkaAWH7dmX1JrI"

# Choose the LLM that will drive the agent
# llm = Fireworks()
llm = ChatOpenAI()

# Construct the Self Ask With Search Agent
agent = create_self_ask_with_search_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is the weather in Seattle?"})
print(result)
