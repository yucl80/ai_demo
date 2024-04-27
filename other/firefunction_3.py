from pprint import pprint
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from langchain_community.llms import CTransformers
from llama_cpp import Llama
from openai import OpenAI
import json
import time


from transformers import AutoModelForCausalLM, AutoTokenizer

function_spec = [
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
functions = json.dumps(function_spec, indent=4)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to functions. Use them if required. ",
    },
    # {"role": "system", "content": functions},
    {"role": "user", "content": "Hi, can you tell me the current stock price of AAPL and GOOG?"},
]

model_name = "firefunction"

# tokenizer = AutoTokenizer.from_pretrained("fireworks-ai/firefunction-v1")

# model_inputs = tokenizer.apply_chat_template(messages,tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cpu")

# print(tokenizer.convert_ids_to_tokens(model_inputs[0]))


# We should use HF AutoTokenizer instead of llama.cpp's tokenizer because we found that Llama.cpp's tokenizer doesn't give the same result as that from Huggingface. The reason might be in the training, we added new tokens to the tokenizer and Llama.cpp doesn't handle this successfully


before = time.time()
client = OpenAI(api_key="wrTGJ0yeD9BTheuMyeLl6kqic2rG3GtoL3jkaAWH7dmX1JrI",
                base_url="http://127.0.0.1:8000/v1/")
response = client.chat.completions.create(model=model_name, messages=messages, tools=function_spec,
                                          tool_choice="auto", temperature=0.1, response_format={"type": "json_object"})
end = time.time()
print(f"Time taken: {end - before} seconds")


print(response)
