from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

# We should use HF AutoTokenizer instead of llama.cpp's tokenizer because we found that Llama.cpp's tokenizer doesn't give the same result as that from Huggingface. The reason might be in the training, we added new tokens to the tokenizer and Llama.cpp doesn't handle this successfully
llm = Llama.from_pretrained(
    repo_id="meetkai/functionary-small-v2.4-GGUF",
    filename="functionary-small-v2.4.Q4_0.gguf",
    chat_format="functionary-v2",
    tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.4-GGUF"),
    n_gpu_layers=-1
)


messages = [
    {"role": "user", "content": "what's the weather like in  San Francisco, Tokyo, and Paris?"}
]
tools = [ # For functionary-7b-v2 we use "tools"; for functionary-7b-v1.4 we use "functions" = [{"name": "get_current_weather", "description":..., "parameters": ....}]
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
                        "description": "The city and state, e.g., San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
import time
start_time = time.time()
result = llm.create_chat_completion(
      messages = messages,
      tools=tools,
      tool_choice="auto",
)
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))

print(result["choices"][0]["message"])

start_time = time.time()
result = llm.create_chat_completion(
      messages = messages,
      tools=tools,
      tool_choice="auto",
)
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))