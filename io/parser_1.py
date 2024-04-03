from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts.chat import ChatPromptTemplate
output_parser = CommaSeparatedListOutputParser()
result = output_parser.parse("hi, bye")
print(type(result))
print(result)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

result1 = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
print(type(result1))
print(result1)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
        model_name="glm-3-turbo",
#        model_name = "glm-4",
#        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key="YOUR_API_KEY",
        streaming=False,
        temperature=0.01,
        timeout= 180,
    ) 
template = "Generate a list of 5 {text}. use RGB express colors \n\n{format_instructions}"

chat_prompt = ChatPromptTemplate.from_template(template)
chat_prompt = chat_prompt.partial(format_instructions= (
            "Your response should be a list of comma separated values, "
            "eg: `red(255,0,0),blue(0,0,255),green(0,255,0)`"
        ))
chain = chat_prompt | llm | output_parser
obj = chain.invoke({"text": "colors"})

print(type(obj))
print(obj)