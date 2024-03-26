from langchain_community.embeddings import LlamaCppEmbeddings
llama = LlamaCppEmbeddings(model_path="/home/test/.cache/ggufs/bge-large-zh-v1.5-q4_k_m.gguf")
text = "This is a test document."
query_result = llama.embed_query(text)
doc_result = llama.embed_documents([text])
print(query_result)
print(doc_result)

