import llama_cpp

llm = llama_cpp.Llama(model_path="/home/test/llm-models/bge-large-zh-v1.5-q4_k_m.gguf", embedding=True)

embeddings = llm.create_embedding("Hello, world!")

# or create multiple embeddings at once

embeddings = llm.create_embedding(["Hello, world!", "Goodbye, world!"])