from langchain_community.vectorstores.redis import Redis
from langchain_openai import OpenAIEmbeddings

vector_store = Redis(
    redis_url="redis://192.168.85.130:6379",
    embedding=OpenAIEmbeddings(base_url="http://localhost:8000/v1",api_key="nokey"),
    index_name="users",
)

from langchain_core.documents import Document

document_1 = Document(page_content="foo", metadata={"baz": "bar"})
document_2 = Document(page_content="thud", metadata={"bar": "baz"})
document_3 = Document(page_content="i will be deleted :(")

documents = [document_1, document_2, document_3]
ids = ["1", "2", "3"]
vector_store.add_documents(documents=documents, ids=ids)

results = vector_store.similarity_search(query="thud",k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
