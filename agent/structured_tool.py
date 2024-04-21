from typing import List

from langchain_core.tools import tool


@tool
def get_data(n: int) -> List[dict]:
    """Get n datapoints."""
    return [{"name": "foo", "value": "bar"}] * n


tools = [get_data]

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

# Get the prompt to use - you can modify this!
# If you want to see the prompt in full, you can at: https://smith.langchain.com/hub/hwchase17/openai-functions-agent
prompt = hub.pull("hwchase17/openai-functions-agent")

llm = ChatOpenAI(temperature=0, base_url="http://127.0.0.1:8000/v1/", api_key="APIKEY")

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

for chunk in agent_executor.stream({"input": "get me three datapoints"}):
    # Agent Action
    if "actions" in chunk:
        for action in chunk["actions"]:
            print(
                f"Calling Tool ```{action.tool}``` with input ```{action.tool_input}```"
            )
    # Observation
    elif "steps" in chunk:
        for step in chunk["steps"]:
            print(f"Got result: ```{step.observation}```")
