from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import os
os.environ["TAVILY_API_KEY"]="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"

tools = [TavilySearchResults(max_results=2)]

#search = TavilySearchAPIWrapper(tavily_api_key="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")
#tavily_tool = TavilySearchResults(api_wrapper=search,max_results=3)
#tools = [tavily_tool]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")

# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(base_url = "http://127.0.0.1:8000/v1/",api_key="APIKEY", temperature=0)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "什么是有效数？"})

from langchain_core.messages import AIMessage, HumanMessage

#agent_executor.invoke(
#    {
#        "input": "what's my name? Don't use tools to look this up unless you NEED to",
#        "chat_history": [
#            HumanMessage(content="hi! my name is bob"),
#            AIMessage(content="Hello Bob! How can I assist you today?"),
#        ],
#    }
#)