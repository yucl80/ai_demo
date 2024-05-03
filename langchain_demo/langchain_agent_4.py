from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from yucl.utils import create_llm
from langchain_experimental.tools import PythonREPLTool
from langchain_core.callbacks import StdOutCallbackHandler, BaseCallbackHandler,AsyncCallbackManagerForToolRun
from langchain.chains import LLMChain
from langchain.agents import AgentType, initialize_agent, load_tools

llm = create_llm(model="llama-3-8b")
tools = [TavilySearchResults(max_results=1), PythonREPLTool()]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

handler = StdOutCallbackHandler()

# Construct the Tools agent
#agent = create_tool_calling_agent(llm, tools, prompt)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)





# config = {"callbacks": [MyCustomHandler()]}

result = agent.run({"input": "what is LangChain?"})
print(result)

#result = agent_executor.invoke({"input": "what is LangChain?"}, config=config)
#print(result)
