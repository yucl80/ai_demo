from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from prompt_loader import load_prompt_from_file

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [tool]

# Get the prompt to use - you can modify this!
# If you want to see the prompt in full, you can at: https://smith.langchain.com/hub/hwchase17/openai-functions-agent
#prompt = hub.pull("hwchase17/openai-functions-agent")

file_path = "/home/test/src/code/prompts/hwchase17/openai-functions-agent.json"
prompt = load_prompt_from_file(file_path)
print(prompt) 

llm = ChatOpenAI(temperature=0, base_url = "http://127.0.0.1:8000/v1/",api_key="APIKEY")

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)

response = agent_executor.invoke({"input": "What is Leo DiCaprio's middle name?"})

print(response)