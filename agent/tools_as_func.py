from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0, base_url = "http://127.0.0.1:8000/v1/",api_key="APIKEY")

tools = [MoveFileTool()]
functions = [convert_to_openai_function(t) for t in tools]

message = model.invoke(
    [HumanMessage(content="move file foo to bar")], functions=functions
)
print(message)
