from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from yucl.utils import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loaders = [
    TextLoader("/home/test/src/langchain/docs/docs/modules/paul_graham_essay.txt"),
    TextLoader("/home/test/src/langchain/docs/docs/modules/state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)

sub_docs = vectorstore.similarity_search("justice breyer")

print(sub_docs[0].page_content)

retrieved_docs = retriever.get_relevant_documents("justice breyer")

print(len(retrieved_docs[0].page_content))

# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

print(len(list(store.yield_keys())))

sub_docs = vectorstore.similarity_search("justice breyer")
print(sub_docs[0].page_content)

retrieved_docs = retriever.get_relevant_documents("justice breyer")
print(len(retrieved_docs[0].page_content))

print(retrieved_docs[0].page_content)