#https://blog.csdn.net/yuanmintao/article/details/136268609?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from yucl.utils import get_api_key
from ChatGLM4 import ChatZhipuAI
from langchain_openai import ChatOpenAI
import langchain
import os
#langchain.debug = True

current_process_id = os.getpid()
print(current_process_id)

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    current_directory = os.getcwd()
    print(current_directory)
    current_process_id = os.getpid()
    print(current_process_id)
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

print(multiply.name)
print(multiply.description)
print(multiply.args)

pythonREPLTool = PythonREPLTool()
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

from langchain.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

from langchain import hub
prompt = hub.pull("hwchase17/structured-chat-agent")
prompt.pretty_print()

#定义工具
tools = [pythonREPLTool, wikipedia, search, multiply, add, exponentiate]
from langchain.agents import create_structured_chat_agent
from yucl.utils import get_api_key,get_api_token


base_url = "http://127.0.0.1:8000/v1/"
#base_url = "https://open.bigmodel.cn/api/paas/v4/"

llm = ChatOpenAI(api_key=get_api_token(), base_url=base_url,model="glm-4")

# 创建 structured chat agent
agent = create_structured_chat_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
# 传入agent和tools来创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

rep = agent_executor.invoke(
    {
        "input": "12加13等于多少？"
    }
)

print(rep)

#rep = agent_executor.invoke(
#    {
#        "input": "中国平安现在的股票价格是多少？"
#    }
#)
#print(rep)

#待排序的customer_list 
customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
rep = agent_executor.invoke(
    {
        "input": f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}"""
    }
)
print(rep)