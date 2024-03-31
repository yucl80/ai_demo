from llama_cpp import Llama
llm = Llama(model_path="/home/test/llm-models/llama-2-13b-chat.Q4_K_M.gguf", chat_format="chatml-function-calling")
rep = llm.create_chat_completion(
    messages = [
        {
          "role": "system",
          "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"

        },
        {
          "role": "user",
          "content": "Extract Jason is 25 years old"
        }
      ],
      tools=[{
        "type": "function",
        "function": {
          "name": "UserDetail",
          "parameters": {
            "type": "object",
            "title": "UserDetail",
            "properties": {
              "name": {
                "title": "Name",
                "type": "string"
              },
              "age": {
                "title": "Age",
                "type": "integer"
              }
            },
            "required": [ "name", "age" ]
          }
        }
      }],
      tool_choice=[{
        "type": "function",
        "function": {
          "name": "UserDetail"
        }
      }]
)

print(rep)