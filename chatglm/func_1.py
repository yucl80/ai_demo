from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from code.deploy.yucl_utils.jwt_token import get_api_token

# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")
    
llm = ChatOpenAI(
        model_name="glm-3-turbo",
#        model_name = "glm-4",
#       openai_api_base="https://open.bigmodel.cn/api/paas/v4",
       openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 120,
    ) 

llm_with_tools = llm.bind_tools([Multiply])

from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

tool_chain = llm_with_tools | JsonOutputToolsParser()

rep = tool_chain.invoke("what's 3 * 12")
print(rep)