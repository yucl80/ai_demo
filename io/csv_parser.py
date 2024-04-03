from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(base_url="http://localhost:8000/v1", openai_api_key="your_api_key",temperature=0)

chain = prompt | model | output_parser

result =chain.invoke({"subject": "ice cream flavors"})

print(result)