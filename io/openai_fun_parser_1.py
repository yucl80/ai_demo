from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


openai_functions = [convert_to_openai_function(Joke)]

from jwt_token import get_api_key, get_api_token
from pprint import pprint
model = ChatOpenAI(
#        model_name="glm-3-turbo",
        model_name = "glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
#        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    )

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant"), ("user", "{input}")]
)

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

def print_output(output) -> any:
    print("Output Beings:----")
    print(output)
    print("Output Ends----")
    return output

parser = JsonOutputFunctionsParser()
chain = prompt | model.bind(functions=openai_functions) | print_output|parser
result = chain.invoke({"input": "tell me a joke"})
pprint(result)