from llama_cpp import Llama


# You are an AI programming assistant, utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
# ### Instruction: <<function>>[{"name": "function_name", "description": "Description", "parameters": {...}}, ...]
# <<question>>{prompt}
# ### Response:

llm = Llama(model_path="/home/test/llm-models/gorilla-openfunctions-v2-q4_K_M.gguf", n_gpu_layers=0, n_ctx=16384, temperature=0.0, repeat_penalty=1.1)
print(llm.create_chat_completion(
      messages = [
        {
          "role": "user",
          "content": "What's the weather like in Oslo?"
        }
      ],
      tools=[{
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": [ "celsius", "fahrenheit" ]
              }
            },
            "required": [ "location" ]
          }
        }
      }],
      temperature=0.0,
))
