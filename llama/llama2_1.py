import llama_cpp

llm = llama_cpp.Llama(model_path="/home/test/src/llama-2-13b-chat.Q4_K_M.gguf", embedding=True)

embeddings = llm.create_embedding("Hello, world!")

# or create multiple embeddings at once

embeddings = llm.create_embedding(["Hello, world!", "Goodbye, world!"])