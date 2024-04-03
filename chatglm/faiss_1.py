from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings,HuggingFaceInferenceAPIEmbeddings

from yucl.utils import get_api_token

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

embedding = model.embed_documents("hi this is harrison")
print(embedding)


db = FAISS.from_documents(docs, model)
print(db.index.ntotal)