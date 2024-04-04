from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr
from yucl.utils import HttpEmbeddingsClient
from yucl.utils import log_output
#print(pages[0])


# loader = PyPDFLoader("/home/test/src/code/retrieval/pdf/创业板 ETF 期权合约基本条款.pdf")
# pages = loader.load_and_split()
    
embeddings = HttpEmbeddingsClient()

# faiss_index = FAISS.from_documents(pages,embeddings)
# docs = faiss_index.similarity_search("交割方式是什么?", k=2)
# for doc in docs:
#    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

# loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
# pages = loader.load()
# print(pages[4].page_content)



# from langchain_community.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader("/home/test/src/code/retrieval/pdf/创业板 ETF 期权合约基本条款.pdf")
# data = loader.load()
# print(data)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
def test_pdf_loader():
    loader = PyMuPDFLoader("/home/test/src/code/retrieval/pdf/创业板 ETF 期权合约基本条款.pdf")
    data = loader.load()
    print(data)

from langchain_community.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader("/home/test/src/code/retrieval/pdf/")
docs = loader.load()
print(docs[0])