from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import pprint

def print_output(output) -> any:
    print("Output Beings:----")
    print(output)
    print("Output Ends----")
    return output

response_schemas = [
  ResponseSchema(name="domainList", description="description to model with the example: [{domain_name: string, domain_description: string}]", 
  type="array(objects)")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

format_instructions = """The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{ "domainList":[
{
        "domain_name": string  // 业务域名称
        "domain_description": string  // 业务域描述.
}]}
"""

prompt = PromptTemplate(
    template="你是一个金融领域资深业务架构师，'请尽可能全面的回答用户的问题.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(model="chatglm3-q8" ,base_url="http://localhost:8000/v1", openai_api_key="your_api_key",temperature=0)
chain = prompt | model | print_output | output_parser

result = chain.invoke({"question": "证券行业包含哪些业务域? 请遵循MECE原则。不重复不遗漏的列出所有业务域。Reply using JSON format"})

print(pprint.pformat(result))