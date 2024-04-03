from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

#parser.parse(misformatted)

from langchain.output_parsers import OutputFixingParser
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

new_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

result = new_parser.parse(misformatted)
print(result)