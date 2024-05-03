
from transformers import PreTrainedTokenizerFast, BertTokenizer
from transformers import (AutoModelForCausalLM, LlamaForCausalLM,
                          LlamaTokenizerFast)
tokenizer = LlamaTokenizerFast.from_pretrained(
    "meetkai/functionary-small-v2.4-GGUF", legacy=False)



# print("begin -----------------")
# print(tokenizer)
# print("end ------------------")

# print(tokenizer.eos_token)
# print(tokenizer.chat_template)

# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file="/home/test/.cache/huggingface/hub/models--meetkai--functionary-small-v2.4-GGUF/snapshots/a0d171eb78e02a58858c464e278234afbcf85c5c/tokenizer.json")

messages=[
    {'role': 'system', 'content': 'You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.'}, 
    {'role': 'user', 'content': 'what is LangChain?'},
    {'role': 'assistant', 'content': None, 
        'tool_calls': [{'id': 'call_C4df2xd6obyxQeeegMj0otNM', 'type': 'function', 'function': {'name': 'tavily_search_results_json', 'arguments': '"{\\"query\\": \\"\\\\\\"LangChain definition\\\\\\"\\"}"'}
                        }]
    }]

template = tokenizer.apply_chat_template(
    messages)
print(template)
#print(fast_tokenizer)

print(tokenizer.convert_ids_to_tokens(template))

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("meetkai/functionary-small-v2.4-GGUF", padding_side="left")
# model_inputs = tokenizer(["system: you are a bot \n  user: A list of colors: red, blue"], return_tensors="pt")
# print(model_inputs)


from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenized_chat)
print(tokenizer.decode(tokenized_chat[0]))


from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained(
    'fireworks-ai/firefunction-v1', use_fast=True
)
print(tokenizer.chat_template)

function_spec = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol, e.g. AAPL, GOOG"
                }
            },
            "required": [
                "symbol"
            ]
        }
    },
    {
        "name": "check_word_anagram",
        "description": "Check if two words are anagrams of each other",
        "parameters": {
            "type": "object",
            "properties": {
                "word1": {
                    "type": "string",
                    "description": "The first word"
                },
                "word2": {
                    "type": "string",
                    "description": "The second word"
                }
            },
            "required": [
                "word1",
                "word2"
            ]
        }
    }
]
functions = json.dumps(function_spec, indent=4)

messages = [
    {'role': 'functions', 'content': functions},
    {'role': 'system', 'content': 'You are a helpful assistant with access to functions. Use them if required.'},
    {'role': 'user', 'content': 'Hi, can you tell me the current stock price of AAPL?'}
]

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
# print(tokenized_chat)
# print(tokenizer.decode(tokenized_chat[0]))


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("fireworks-ai/firefunction-v1")

chat = [
  {"role": "functions", "content": functions},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

test = tokenizer.apply_chat_template(chat, tokenize=False)
print(tokenizer.chat_template)
print(tokenizer)