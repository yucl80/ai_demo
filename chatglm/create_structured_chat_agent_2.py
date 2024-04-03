import time
import langchain
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.agents.format_scratchpad.openai_tools import \
    format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import \
    OpenAIToolsAgentOutputParser
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
#langchain.debug = True
from yucl.utils import create_llm


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    print("call get_word_length")   
    return len(word)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are a helpful assistant ,请调用工具来回答用户的提问",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

pythonREPLTool = PythonREPLTool()

#prompt.pretty_print()
llm = create_llm()

#定义工具
tools = [ get_word_length]

llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


# 传入agent和tools来创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

#rep = agent_executor.invoke(  { "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"   })
st = time.perf_counter()
rep = agent_executor.invoke(  { "input": "请输出当前系统的时间" })
et = time.perf_counter() - st
print("search time:", et)
print(rep)

l = list(agent_executor.stream({"input": "How many letters in the word eudca"}))
print(l)



