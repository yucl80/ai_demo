from openai import OpenAI
import json


base_url = "http://127.0.0.1:8000/v1/"
#base_url = "https://api.fireworks.ai/inference/v1"
#base_url = "https://open.bigmodel.cn/api/paas/v1/"

import os
os.environ["FIREWORKS_API_KEY"] = "wrTGJ0yeD9BTheuMyeLl6kqic2rG3GtoL3jkaAWH7dmX1JrI"
from langchain_community.llms import Fireworks
client = OpenAI(api_key="wrTGJ0yeD9BTheuMyeLl6kqic2rG3GtoL3jkaAWH7dmX1JrI", base_url=base_url)

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
    {'role': 'system', 'content': 'You are a helpful assistant with access to functions. Use them if required. '},
    {'role': 'user', 'content': 'Hi, can you tell me the current stock price of AAPL?'}
]

model_name = "accounts/fireworks/models/firefunction-v1"

response = client.chat.completions.create(model=model_name, messages=messages,temperature=0.1,tool_choice="auto",response_format={"type": "json_object"});

print(response)

from pprint import pprint

print(response.choices[0].message)