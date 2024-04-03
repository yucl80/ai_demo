from yucl.utils import get_api_key,get_api_token
import os
os.environ["OPENAI_API_KEY"] = get_api_token()

# Load docs
from langchain_community.document_loaders.web_base import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# Store splits
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# RAG prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

# LLM
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

#llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_base="https://open.bigmodel.cn/api/paas/v4",   temperature=0)
llm = ChatOpenAI(
        model_name="glm-3-turbo",
#        model_name = "glm-4",
#        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_base="http://127.0.0.1:8000/v1/",
        openai_api_key=get_api_token(),
        streaming=False,
        temperature=0.01,
        timeout= 120,
    )    
# RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
question = "What are the approaches to Task Decomposition?"
result = qa_chain({"query": question})
print(result["result"])