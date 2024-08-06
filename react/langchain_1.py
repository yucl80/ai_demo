import openai
import os
# from langchain.llms import OpenAI
from langchain_openai import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent,create_react_agent
from dotenv import load_dotenv
from langchain_core.tools import tool
load_dotenv()

@tool
def get_current_weather(location: str,) -> str:
    """Get the current weather in a given location."""
    print("call get_current_weather")
    return f"The weather in {location} is sunny today."

os.environ["SERPER_API_KEY"] ="cd123db7d54b206a3a109e205ae231d90f103902"
os.environ["https_proxy"]="http://127.0.0.1:49879"
os.environ["http_proxy"]="http://127.0.0.1:49879"


llm = OpenAI(model_name="text-davinci-003" ,temperature=0,base_url="http://localhost:8000/v1",api_key="nokey")
tools = load_tools(["google-serper", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# agent = create_react_agent(llm=llm,tools=tools,verbose=True)

result = agent.invoke("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
print(result)