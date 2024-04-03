from langchain_core.pydantic_v1 import BaseModel, Field
from yucl_utils import ChatOpenAI

# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


llm = ChatOpenAI() 

llm_with_tools = llm.bind_tools([Multiply])

from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

tool_chain = llm_with_tools | JsonOutputToolsParser()

rep = tool_chain.invoke("what's 3 * 12")
print(rep)