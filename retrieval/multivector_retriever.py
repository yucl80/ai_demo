from langchain.retrievers.multi_vector import MultiVectorRetriever

from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from yucl.utils import create_llm,HttpEmbeddingsClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint

loaders = [
    TextLoader("/home/test/src/langchain/docs/docs/modules/paul_graham_essay.txt"),
    TextLoader("/home/test/src/langchain/docs/docs/modules/state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=HttpEmbeddingsClient()
)
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
import uuid

doc_ids = [str(uuid.uuid4()) for _ in docs]

# The splitter to use to create smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)
    
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# Vectorstore alone retrieves the small chunks
results = retriever.vectorstore.similarity_search("justice breyer")[0]
pprint(results)
