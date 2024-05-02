import json
from openai import OpenAI
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'gorilla-llm/gorilla-openfunctions-v2', use_fast=True
)

print(tokenizer.chat_template)

query = "What's the weather like in the two cities of Boston and San Francisco?"
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    }


]


client = OpenAI(api_key="YOUR_API_KEY", base_url="http://127.0.0.1:8000/v1")

completion = client.chat.completions.create(
    model="gorilla-openfunctions-v2",
    temperature=0.0,
    messages=[{"role": "user", "content": query}],
    # functions=functions,
    tools=functions,
    tool_choice={
        "type": "function",
        "function": {
          "name": "get_current_weather"
        }
      }
)
print(completion)
