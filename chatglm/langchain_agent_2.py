# https://zhuanlan.zhihu.com/p/679421001

from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI
from code.deploy.yucl_utils.jwt_token import get_api_token
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core import utils
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import  format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import tool
from langchain_experimental.tools import PythonREPLTool
from tool_register import dispatch_tool,register_tool

def get_glm(temprature):
    llm = ChatOpenAI(
        model_name="glm-3-turbo",
#        model_name = "glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
#        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=temprature,
        timeout= 60,
    )
    
    return llm


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

pythonREPLTool = PythonREPLTool()

tools = [pythonREPLTool, get_word_length]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = get_glm(0.01).bind(tools=[utils.function_calling.convert_to_openai_tool(tool) for tool in tools])


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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

rep1 = agent_executor.invoke({"input": "How many letters in the word eudca"})
register_tool(tools)
dispatch_tool(rep1)

#rep1 = list(agent_executor.stream({"input": "How many letters in the word eudca"}))
print(rep1)

#result = json.dumps({"price": 12412}, ensure_ascii=False)
#response, history = model.chat(tokenizer, result, history=history, role="observation")
#print(response)

