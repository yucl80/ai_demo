from llama_cpp import Llama
llm = Llama(
      model_path="/home/test/models/llama-2-13b-chat.Q4_K_M.gguf",
      chat_format="llama-2",
      n_threads= 12,
)
rep = llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly describes images."},
          {
              "role": "user",
              "content": "Describe this image in detail please."
          }
      ]
)

print(rep)