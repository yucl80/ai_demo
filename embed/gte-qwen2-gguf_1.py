
from langchain_community.embeddings import LlamaCppEmbeddings
# llama = LlamaCppEmbeddings(model_path="D:\\llm\\LMStudio\\BAAI\\bge-m3-gguf\\bge-m3-q8_0.gguf",verbose=False)
llama = LlamaCppEmbeddings(model_path="D:\\llm\\LMStudio\\lmstudio-community\\Qwen\\gte-Qwen2-1.5B-instruct-Q4_K_M.gguf",verbose=False)
text = "This is a test document."
# query_result = llama.embed_query(text)
# doc_result = llama.embed_documents([text])
# print(query_result)
# print(doc_result)
texts =["text1"]
import time
begin = time.time()
data = llama.embed_documents(texts)
import numpy as np
print(len(data))
print(len(data[0]))

# data =llama.embed_query("test")
end = time.time()
print("Time elapsed: ", end - begin)
# print(data)

