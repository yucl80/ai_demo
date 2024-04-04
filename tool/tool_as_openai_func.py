from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from yucl.utils import ChatOpenAI
from pprint import pprint

model = ChatOpenAI()

tools = [MoveFileTool()]
functions = [convert_to_openai_function(t) for t in tools]

pprint(functions[0])

message = model.invoke(
    [HumanMessage(content="move file foo to bar")], functions=functions
)

pprint(message)

model_with_functions = model.bind_functions(tools)
result = model_with_functions.invoke([HumanMessage(content="move file foo to bar")])
print(result)



