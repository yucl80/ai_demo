#https://blog.csdn.net/yuanmintao/article/details/136268609?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2
import time

from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool

import langchain

langchain.debug = True
from langchain_community.tools import ShellTool
from yucl.utils import create_llm

from langchain import hub
from langchain.agents import create_openai_functions_agent


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    print("call multiply")
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    print("call add")
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

#print(multiply.name)
#print(multiply.description)
#print(multiply.args)

pythonREPLTool = PythonREPLTool()

shell_tool = ShellTool()


prompt = hub.pull("hwchase17/openai-tools-agent")
print(prompt)
prompt.pretty_print()

#定义工具
tools = [  get_word_length,  multiply, add, exponentiate]


llm = create_llm()
# 创建 structured chat agent
#agent = create_structured_chat_agent(llm, tools, prompt)
agent = create_openai_functions_agent(llm,tools,prompt)

from langchain.agents import AgentExecutor

# 传入agent和tools来创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, max_execution_time=240)


st = time.perf_counter()
#rep = agent_executor.invoke(  { "input": "How many letters in the word eudca"   })
#rep = agent_executor.invoke(  { "input": "请编写python程序实现打印操作系统的当前时间的功能，并执行这个程序"   })
rep = agent_executor.invoke(  { "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"   })
et = time.perf_counter() - st
print("search time:", et)
print(rep)

