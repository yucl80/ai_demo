from langchain_community.embeddings import LlamaCppEmbeddings
llama = LlamaCppEmbeddings(model_path="/home/test/llm-models/bge-large-zh-v1.5-q4_k_m.gguf",verbose=False)
text = "This is a test document."
# query_result = llama.embed_query(text)
# doc_result = llama.embed_documents([text])
# print(query_result)
# print(doc_result)
texts =["text1", "This is a test document.", "text2", "This is another test document."]
import time
begin = time.time()
data = llama.embed_documents(texts)
end = time.time()
print("Time elapsed: ", end - begin)
print(data)

