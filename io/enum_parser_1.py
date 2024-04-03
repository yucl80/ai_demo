from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum
from langchain_openai import ChatOpenAI
from jwt_token import get_api_key, get_api_token
from pprint import pprint
model = ChatOpenAI(
#        model_name="glm-3-turbo",
        model_name = "glm-4",
#        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    )

class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

parser = EnumOutputParser(enum=Colors)

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    """What color eyes does this person have? 

> Person: {person}

Instructions: {instructions}"""
).partial(instructions=parser.get_format_instructions())
chain = prompt | model | parser

chain.invoke({"person": "Frank Sinatra"})