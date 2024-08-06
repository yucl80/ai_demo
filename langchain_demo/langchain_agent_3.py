from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
llm = ChatOpenAI(model="functionary", temperature=0,base_url="http://localhost:8000/v1" ,api_key="NOKEY")
from langchain import hub
from langchain_core.tools import tool
import os

os.environ["TAVILY_API_KEY"]="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"

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

@tool
def get_current_weather(location: str,) -> str:
    """Get the current weather in a given location."""
    print("call get_current_weather")
    return f"The weather in {location} is sunny today."



from langchain.agents import create_openai_tools_agent ,create_json_chat_agent

tools = [get_current_weather,TavilySearchResults(max_results=1)]

agent = create_openai_tools_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,return_intermediate_steps=False,)


# result = agent_executor.invoke({"input":"What's the weather like in the two cities of Boston?"})
result = agent_executor.invoke({"input": "what is LangChain?"})

print(result)

# for step in agent_executor.iter({"input": "what's the weather like in  San Francisco, Tokyo, and Paris?"}):
#     print(step)
#     print("\n\n")
   

