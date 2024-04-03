from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

from jwt_token import get_api_key, get_api_token
from pprint import pprint
model = ChatOpenAI(
#        model_name="glm-3-turbo",
        model_name = "glm-4",
 #       openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    )    
model = model.bind_tools([Joke])

print(model.kwargs["tools"])

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant"), ("user", "{input}")]
)

from langchain.output_parsers.openai_tools import JsonOutputToolsParser

parser = JsonOutputToolsParser()

chain = prompt | model | parser

result = chain.invoke({"input": "tell me a joke"})

print(result)

from typing import List

from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

parser = JsonOutputKeyToolsParser(key_name="Joke")

chain = prompt | model | parser

result = chain.invoke({"input": "tell me a joke"})

print(result)

from langchain.output_parsers.openai_tools import PydanticToolsParser

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


parser = PydanticToolsParser(tools=[Joke])

model = ChatOpenAI(
#        model_name="glm-3-turbo",
        model_name = "glm-4",
 #       openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    ) 

model = model.bind_tools([Joke])
chain = prompt | model | parser

result = chain.invoke({"input": "tell me a joke"})
print(result)
