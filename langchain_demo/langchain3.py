# https://blog.csdn.net/yuanmintao/article/details/136146210

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# embeddings = HuggingFaceEmbeddings()
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loader = WebBaseLoader("https://docs.smith.langchain.com")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


template = """{question}"""
prompt = PromptTemplate.from_template(template)

endpoint_url = "https://open.bigmodel.cn/api/paas/v4"


llm = ChatOpenAI(
    endpoint_url=endpoint_url,
    temperature=0.1,
    api_key="get_api_key",
    model_name="glm-4",
)


# prompt = ChatPromptTemplate.from_messages([
#    ("system", "你是世界级的技术文档作者。"),
#    ("user", "{input}")
# ])

prompt = ChatPromptTemplate.from_template("""仅根据所提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


rep = retrieval_chain.invoke({"input": "langsmith如何帮助测试?"})


print(rep)
