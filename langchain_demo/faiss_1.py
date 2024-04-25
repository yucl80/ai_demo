from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings,HuggingFaceInferenceAPIEmbeddings
from yucl.utils import get_api_token,OpenAIEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.embeddings import LocalAIEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings
import time

loader = TextLoader("/home/test/src/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
#     query_instruction="为这个句子生成表示以用于检索相关文章："
# )
#embeddings = LlamaCppEmbeddings(model_path="/home/test/llm-models/bge-large-zh-v1.5-q4_k_m.gguf",verbose=False)

embeddings = OpenAIEmbeddings( model="bge-large-zh-v1.5")

#embeddings = LocalAIEmbeddings(openai_api_base="http://localhost:8000", model="bge-large-zh-v1.5",openai_api_key="nokey")
# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# print(query_result)
beg = time.time()
db = FAISS.from_documents(docs, embeddings)
end = time.time()
print("Time elapsed: ", end-beg)
print(db.index.ntotal)