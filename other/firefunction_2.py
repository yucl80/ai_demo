from transformers import AutoModelForCausalLM, AutoTokenizer
import json

device = "cpu" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("fireworks-ai/firefunction-v1", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("fireworks-ai/firefunction-v1")

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
    {'role': 'system', 'content': 'You are a helpful assistant with access to functions. Use them if required.'},
    {'role': 'functions', 'content': functions},    
    {'role': 'user', 'content': 'Hi, can you tell me the current stock price of AAPL?'}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

generated_ids = model.generate(model_inputs, max_new_tokens=128)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
