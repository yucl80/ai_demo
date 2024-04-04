import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from yucl.utils import create_llm,HttpEmbeddingsClient
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

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | create_llm()
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=HttpEmbeddingsClient())
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

sub_docs = vectorstore.similarity_search("justice breyer")

pprint(sub_docs)