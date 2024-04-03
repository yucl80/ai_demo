from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from jwt_token import get_api_key, get_api_token
from pprint import pprint
model = ChatOpenAI(
        model_name="glm-3-turbo",
#        model_name = "glm-4",
#       openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 180,
    ) 

# Define your desired data structure.
class BizDomain(BaseModel):   
    name: str = Field(description="business domain name")
    description: str = Field(description="business domain description")

class BizDomainList(BaseModel):
    results: List[BizDomain] = Field(description="list of business domains")

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=BizDomainList)

print(parser.get_format_instructions())

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
#    partial_variables={"format_instructions": "请使用JSON格式返回数据"},
#    partial_variables={"format_instructions": parser.get_format_instructions()},
    partial_variables={"format_instructions": (
            "回答必须是一个JSON数组, "
            "例子: `[{\"name\": \"证券\", \"description\": \"证券业务\"}]`"
        )},
)

chain = prompt | model | parser

result = chain.invoke({"query": "证券行业包含哪些业务域？回答的结果要符合MECE原则。即不重不漏"})
pprint(result)
