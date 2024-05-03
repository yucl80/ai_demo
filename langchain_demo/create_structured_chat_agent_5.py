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
from langchain_core.prompts import PromptTemplate
from yucl.utils import ChatOpenAI, pull_repo
from langchain_core.prompts import ChatPromptTemplate
langchain.debug = True


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    print("call get_word_length")
    return len(word)


# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

pythonREPLTool = PythonREPLTool()
# prompt = pull_repo("hwchase17/structured-chat-agent")
#prompt = pull_repo("hwchase17/structured-chat-agent")
# prompt.pretty_print()

SYSTEM_PROMPT_FOR_CHAT_MODEL = """
    You are an expert in composing functions. You are given a question and a set of possible functions. 
    Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
    If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
    also point it out. You should only return the function call in tool_calls sections.
    """

USER_PROMPT_FOR_CHAT_MODEL = """
    Questions:{user_prompt}\nHere is a list of functions in JSON format that you can invoke:\n{tools}. 
    Should you decide to return the function call(s),Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n
    NO other text MUST be included. 
"""

prompt = ChatPromptTemplate.from_messages(
     [("system", SYSTEM_PROMPT_FOR_CHAT_MODEL),("human", USER_PROMPT_FOR_CHAT_MODEL)]
)

llm = ChatOpenAI(model="openfunctions")


# 定义工具
tools = [get_word_length]


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

input = USER_PROMPT_FOR_CHAT_MODEL.format(
                user_prompt="How many letters in the word eudca?",
                functions=tools,)

rep = agent_executor.invoke({"user_prompt": "How many letters in the word eudca?","tools": tools})
et = time.perf_counter() - st
print("search time:", et)
print(rep)
