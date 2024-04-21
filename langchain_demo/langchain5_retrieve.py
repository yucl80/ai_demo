#https://blog.csdn.net/yuanmintao/article/details/136146210
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.llms.chatglm3 import ChatGLM3


template = """{question}"""
prompt = PromptTemplate.from_template(template)

#llm = ChatZhipuAI(
#   endpoint_url="https://open.bigmodel.cn/api/paas/v4",
#   temperature=0.1,
#   api_key=get_api_key(),
#   model_name="glm-4",
#)

endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

messages = [ ]

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=80000,
    prefix_messages=messages,
    top_p=0.9,
    timeout=60,
)

model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
bgeEmbeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
 )


cache_file = "../ordersample_vector"

vector = any

if os.path.exists(cache_file):
       print("load cache")
       vector = FAISS.load_local(cache_file, bgeEmbeddings,allow_dangerous_deserialization=True)

else:
       print("miss cache")
       loader = CSVLoader(file_path='ordersample.csv')
       data = loader.load()

       from pprint import pprint
       pprint(data)

       text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=1000, chunk_overlap=200, add_start_index=True
       )
  
       all_splits = text_splitter.split_documents(data)

       print(len(all_splits))
 
       vector = FAISS.from_documents(all_splits, bgeEmbeddings)
   
       vector.save_local(cache_file)
   
  
  

retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
rep = retriever.invoke("收货人姓名是张三丰的，有几个订单？金额分别是多少，总共是多少？")
print(rep)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""你是一名数据高级数据分析师,仅根据所提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {question}""")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
retriever_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rep = retriever_chain.invoke("订单ID是123456的收货人是谁，电话是多少?")
print(rep)

rep =retriever_chain.invoke("收货人张三丰有几个订单？金额分别是多少？")
print(rep)

rep =retriever_chain.invoke("李四同有几个订单？金额一共是多少？")
print(rep)
