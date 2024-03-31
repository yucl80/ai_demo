from langchain import hub
from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain_community.chat_models import ChatAnthropic
from langchain_openai import ChatOpenAI
from jwt_token import get_api_key,get_api_token

model = ChatOpenAI(
        model_name="glm-3-turbo",
#        model_name = "glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
 #       openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    )    

#model = ChatAnthropic(model="claude-2")

@tool
def search(query: str) -> str:
    """Search things about current events."""
    return "32 degrees"

tool_list = [search]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/xml-agent-convo")

# Logic for going from intermediate steps to a string to pass into model
# This is pretty tied to the prompt
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    print(log)
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | prompt.partial(tools=convert_tools(tool_list))
    | model.bind(stop=["</tool_input>"])
    | XMLAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)

agent_executor.invoke({"input": "whats the weather in New york?"})