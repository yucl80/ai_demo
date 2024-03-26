#https://blog.csdn.net/yuanmintao/article/details/136268609?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from jwt_token import get_api_key,get_api_token
from ChatGLM4 import ChatZhipuAI
import time
import langchain
#langchain.debug = True
from langchain_openai import ChatOpenAI
from langchain.agents import create_structured_chat_agent
from langchain import hub
from langchain_community.tools import ShellTool

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

print(multiply.name)
print(multiply.description)
print(multiply.args)

pythonREPLTool = PythonREPLTool()

shell_tool = ShellTool()


prompt = hub.pull("hwchase17/structured-chat-agent")
#prompt.pretty_print()

#定义工具
tools = [pythonREPLTool,  get_word_length,  multiply, add, exponentiate]

def get_glm(temprature):
    llm = ChatOpenAI(
        model_name="glm-3-turbo",
#        model_name = "glm-4",
#        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=temprature,
        timeout= 60,
    )
    
    return llm
#llm = get_glm(0.01)
#llm = ChatZhipuAI(
#   endpoint_url="https://127.0.0.1/api/paas/v1/",
#   endpoint_url = "https://open.bigmodel.cn/api/paas/v4"
#   temperature=0.1,
#   api_key=get_api_key(),
#   model_name="glm-3-turbo",
#)

llm = get_glm(0.01)
# 创建 structured chat agent
agent = create_structured_chat_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
# 传入agent和tools来创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

#rep = agent_executor.invoke(  { "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"   })
st = time.perf_counter()
#rep = agent_executor.invoke(  { "input": "How many letters in the word eudca"   })
rep = agent_executor.invoke(  { "input": "打印当前程序的执行目录"   })
et = time.perf_counter() - st
print("search time:", et)
print(rep)

