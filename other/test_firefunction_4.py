
import os

from langchain.chains import LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Optional, Type

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="YOUR_FW_API_KEY",
    model="firefunction",
    temperature=0.0,
    max_tokens=256,
)

math_llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="YOUR_FW_API_KEY",
    model="mixtral-8x7b-instruct",
    temperature=0.0,
)

class CalculatorInput(BaseModel):
    query: str = Field(description="should be a math expression")

class CustomCalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Tool to evaluate mathemetical expressions"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, query: str) -> str:
        """Use the tool."""
        return LLMMathChain(llm=math_llm, verbose=True).run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("not support async")

tools = [
  CustomCalculatorTool()
]

agent = create_openai_tools_agent(llm, tools, prompt)

agent = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent.invoke({"input": "What is the capital of USA?"}))