from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from jwt_token import get_api_key, get_api_token
from pprint import pprint
import json
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
    name: str = Field(description="业务域的名称")
    description: str = Field(description="业务域的描述")

class BizDomainList(BaseModel):
    results: List[BizDomain] = Field(description="list of business domains")

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=BizDomainList)

#print(parser.get_format_instructions())

JSON_FORMAT_INSTRUCTIONS = """输出必须是一个JSON对象 .

```
例子：{{ "result":[{{"name": "证券交易", ", "description": "证券交易是指证券市场上投资者买卖证券的行为。"}}]}}

```

"""

JSON_FORMAT_INSTRUCTIONS2 = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""

def get_format_instructions(pydantic_object) -> str:  
            # Copy schema to avoid altering original Pydantic schema.
    schema = {k: v for k, v in pydantic_object.schema().items()}
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]
       # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)
    return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)

print(get_format_instructions(BizDomainList))
        
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": get_format_instructions(BizDomainList)},
  
)

chain = prompt | model | parser

result = chain.invoke({"query": "证券行业包含哪些业务域？回答的结果要符合MECE原则。即不重复不遗漏"})
pprint(result)
