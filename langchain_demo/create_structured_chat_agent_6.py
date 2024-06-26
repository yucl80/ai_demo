# https://blog.csdn.net/yuanmintao/article/details/136268609?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-136268609-blog-136146210.235%5Ev43%5Epc_blog_bottom_relevance_base2
from langchain.agents import AgentExecutor
import time

from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool


import langchain
from langchain import hub

# from langchain_community.tools import ShellTooll
from langchain.agents import AgentType, create_structured_chat_agent, initialize_agent
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from yucl.utils import ChatOpenAI, pull_repo

langchain.debug = True


@tool
def get_current_weather(location: str,) -> str:
    """Get the current weather in a given location."""
    print("call get_current_weather")
    return "The weather in {location} is sunny today."


# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

pythonREPLTool = PythonREPLTool()
# prompt = pull_repo("hwchase17/structured-chat-agent")
prompt = pull_repo("hwchase17/structured-chat-agent")
# prompt.pretty_print()

llm = ChatOpenAI(model="openfunctions")


# 定义工具
tools = [get_current_weather]


# 创建 structured chat agent
agent = create_structured_chat_agent(llm, tools, prompt)


# 传入agent和tools来创建Agent执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    return_intermediate_steps=True,
)

# rep = agent_executor.invoke(  { "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"   })
st = time.perf_counter()
rep = agent_executor.invoke({"input": "What's the weather like in the two cities of Boston"})
et = time.perf_counter() - st
print("search time:", et)
print(rep)
