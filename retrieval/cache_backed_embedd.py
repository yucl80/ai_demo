from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from yucl.utils import HttpEmbeddingsClient
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings

underlying_embeddings = HttpEmbeddingsClient()

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace="bge-m3"
)

print(list(store.yield_keys()))

raw_documents = TextLoader("/home/test/src/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

db = FAISS.from_documents(documents, cached_embedder)

# query = "What did the president say about Ketanji Brown Jackson"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

# embedding_vector = HttpEmbeddingsClient().embed_query(query)
# docs = db.similarity_search_by_vector(embedding_vector)
# print(docs[0].page_content)

retriever = db.as_retriever()
#retriever = db.as_retriever(search_type="mmr")
retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.01}
)
retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
print(len(docs))
