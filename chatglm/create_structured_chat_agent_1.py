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
#from langchain_community.tools import ShellTooll
from langchain.agents import AgentType, initialize_agent
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    print("call get_word_length")   
    return len(word)

#print(multiply.name)
#print(multiply.description)
#print(multiply.args)

pythonREPLTool = PythonREPLTool()

prompt = hub.pull("hwchase17/structured-chat-agent")
#prompt.pretty_print()

llm = ChatOpenAI(
        model_name="glm-3-turbo",
#        model_name = "glm-4",
#        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    )    
   

#定义工具
tools = [ get_word_length]


# 创建 structured chat agent
agent = create_structured_chat_agent(llm, tools, prompt)


from langchain.agents import AgentExecutor
# 传入agent和tools来创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,return_intermediate_steps=True)

#rep = agent_executor.invoke(  { "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"   })
st = time.perf_counter()
rep = agent_executor.invoke(  { "input": "How many letters in the word eudca" })
et = time.perf_counter() - st
print("search time:", et)
print(rep)

