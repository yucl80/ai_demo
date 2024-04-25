#https://blog.csdn.net/yuanmintao/article/details/136146210
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage
from yucl.utils import get_api_key
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


template = """{question}"""
prompt = PromptTemplate.from_template(template)

endpoint_url = "http://127.0.0.1:8000/v1"


llm = ChatOpenAI(
   base_url=endpoint_url,
   temperature=0.1,
   api_key=get_api_key(),
   model="llama-3-8b",
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术文档作者。"),
    ("user", "{input}")
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
rep = chain.invoke({"input": "langsmith如何帮助测试?"})


print(rep)