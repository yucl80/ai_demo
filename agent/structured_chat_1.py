from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from yucl.utils import ChatOpenAI

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/structured-chat-agent")

# Choose the LLM that will drive the agent
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

# Construct the JSON agent
agent = create_structured_chat_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

result = agent_executor.invoke({"input": "what is LangChain?"})

print(result)