from langchain_community.embeddings import LlamaCppEmbeddings
# llama = LlamaCppEmbeddings(model_path="/home/test/llm-models/bge-large-zh-v1.5-q4_k_m.gguf",verbose=False)
llama = LlamaCppEmbeddings(model_path="D:\\llm\\LMStudio\\lmstudio-community\\Qwen\\stella_en_1.5B_v5.gguf",verbose=True)
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

