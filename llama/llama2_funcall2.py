from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
#llm = Llama(model_path="/home/test/src/llama.cpp/models/llama-2-13b-chat.Q4_K_M.gguf", chat_format="chatml-function-calling")

tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v3.2-GGUF")
llm = Llama.from_pretrained( 
  filename="functionary-small-v3.2.Q4_0.gguf",
  chat_format="functionary-v2",
  tokenizer=tokenizer,
  repo_id="meetkai/functionary-small-v3.2-GGUF"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)

# messages = [{"role": "user", "content": "What is the weather in Istanbul and Singapore respectively?"}]

# final_prompt = tokenizer.apply_chat_template(messages, tools, add_generation_prompt=True, tokenize=False)

# inputs = tokenizer(final_prompt, return_tensors="pt")

# pred = llm.generate_tool_use(**inputs, max_new_tokens=128, tokenizer=tokenizer)
# print(tokenizer.decode(pred.cpu()[0]))

output = llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly describes images."},
          {
              "role": "user",
              "content": "Describe this image in detail please."
          }
      ]
)
print(output)

output = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",
    },
    temperature=0.7,
)
print(output)

output =llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"team_name": {"type": "string"}},
            "required": ["team_name"],
        },
    },
    temperature=0.7,
)
print(output)