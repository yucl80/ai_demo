from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


llm = ChatOpenAI(model="chatglm3",base_url="http://localhost:8000/v1", api_key="YOUR_API_KEY") 

llm_with_tools = llm.bind_tools([Multiply])

from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

response = llm_with_tools.invoke("what's 3 * 12")
print(response)

tool_chain = llm_with_tools | JsonOutputToolsParser()
import time
begin = time.time()
rep = tool_chain.invoke("what's 3 * 10")
end = time.time() 

print(f"Time taken: {end - begin} seconds")
print(rep)