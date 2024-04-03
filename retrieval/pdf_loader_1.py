from langchain_community.document_loaders import PyPDFLoader
from typing import List, Optional
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr
loader = PyPDFLoader("/home/test/src/langchain/docs/docs/integrations/document_loaders/example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()
from yucl_utils import HttpEmbeddingsClient
from yucl_utils import log_output
#print(pages[0])

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
#from langchain_openai import OpenAIEmbeddings

log_output("asasfdasdfsf")
    
embeddings = HttpEmbeddingsClient()
results = embeddings.embed_documents(["How will the community be engaged?"])


#faiss_index = FAISS.from_documents(pages,embeddings)
#docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
#for doc in docs:
#   print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
    