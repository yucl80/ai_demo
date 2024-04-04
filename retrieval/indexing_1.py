from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
#from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from yucl.utils import OpenAIEmbeddings
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

collection_name = "test_index"

embedding = OpenAIEmbeddings()

#vectorstore = FAISS.from_texts([""], embedding)
vectorstore = FAISS( embedding_function= embedding,  index=IndexFlatL2(1024), docstore=InMemoryDocstore(), index_to_docstore_id={},)


namespace = f"faiss2/{collection_name}"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_cache.sql"
)

record_manager.create_schema()

doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})

def _clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")
    

results = index([doc1, doc1], record_manager, vectorstore, cleanup=None, source_id_key="source")

print(results)

_clear()

results = index([doc1, doc2], record_manager, vectorstore, cleanup=None, source_id_key="source")

print(results)