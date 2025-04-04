from langchain.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from yucl.utils import OpenAIEmbeddings
from langchain import PromptTemplate

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import sys
import os

from langchain.chains import NebulaGraphQAChain
from langchain_community.graphs import NebulaGraph
from langchain_openai import ChatOpenAI

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# load the pdf and split it into chunks
#loader = OnlinePDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf")
loader = PyPDFLoader("/home/test/src/601318_20240322_QFI4.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

with SuppressStdout():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    # Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    llm = ChatOpenAI(model="chatglm3-q8",base_url="http://localhost:8000/v1", api_key="YOUR_API_KEY", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])) 

    # llm = Ollama(model="llama2:13b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})
    print(result)