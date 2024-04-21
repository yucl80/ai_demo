from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from yucl.utils import get_api_key,get_api_token

import time
import langchain
langchain.debug = True
from langchain_openai import ChatOpenAI
from langchain.agents import create_structured_chat_agent,create_openai_tools_agent
from langchain import hub
from langchain_community.tools import ShellTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


prompt = hub.pull("hwchase17/structured-chat-agent")

p1 = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }    | prompt

print(p1)