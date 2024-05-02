from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("fireworks-ai/firefunction-v1")
#tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
import json

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
     {"role": "functions", "content": functions},
  {"role": "system", "content": "You are a helpful assistant with access to functions.\n Use them if required."},
 
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

data = tokenizer.apply_chat_template(messages, tokenize=False)
print(tokenizer.chat_template)
print(tokenizer)
print(data)

