import time
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

# We should use HF AutoTokenizer instead of llama.cpp's tokenizer because we found that Llama.cpp's tokenizer doesn't give the same result as that from Huggingface. The reason might be in the training, we added new tokens to the tokenizer and Llama.cpp doesn't handle this successfully
llm = Llama.from_pretrained(
    # repo_id="brittlewis12/Octopus-v2-GGUF",
    # filename="octopus-v2.Q4_K_M.gguf",
    repo_id="meetkai/functionary-medium-v2.2-GGUF",
    filename="functionary-medium-v2.2.q4_0.gguf",
    #  chat_format="firefunction-v1",
    #   tokenizer=LlamaHFTokenizer.from_pretrained("brittlewis12/Octopus-v2-GGUF"),
    #  n_gpu_layers=-1
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to functions. Use them if required.",
    },
    {"role": "user", "content": "Hi, can you tell me the current stock price of AAPL?"},   
]
# mes
tools = [  # For functionary-7b-v2 we use "tools"; for functionary-7b-v1.4 we use "functions" = [{"name": "get_current_weather", "description":..., "parameters": ....}]
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
                            "description": "The city and state, e.g., San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
        },
    },
    {
        "type": "function",
        "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock symbol, e.g. AAPL, GOOG",
                        }
                    },
                    "required": ["symbol"],
                },
        },
    },
    {
        "type": "function",
        "function": {
                "name": "check_word_anagram",
                "description": "Check if two words are anagrams of each other",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word1": {"type": "string", "description": "The first word"},
                        "word2": {"type": "string", "description": "The second word"},
                    },
                    "required": ["word1", "word2"],
                },
        },
    },
]
start_time = time.time()
result = llm.create_chat_completion(
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))

print(result["choices"][0]["message"])
