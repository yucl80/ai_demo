from openai import OpenAI
import json
import time

base_url = "http://localhost:8000/v1/"


client = OpenAI(base_url=base_url, api_key="YOUR_API_KEY")


def function_chat1():
    function_spec = [
        {
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
        {
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
    ]
    functions = json.dumps(function_spec, indent=4)

    messages = [
        # {"role": "functions", "content": functions},
        {
            "role": "system",
            "content": "You are a helpful assistant with access to functions. Use them if required.",
        },
        {
            "role": "user",
            "content": "Hi, can you tell me the current stock price of AAPL?",
        },
    ]

    response = client.chat.completions.create(
        model="Phi-3-mini-128k-directml-int4-awq-block-128-onnx",
        messages=messages,
        tool_choice="auto",
        stream=False,
    )
    for r in response:
        print(r)


def function_chat():
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco and CA?",
        },
    ]
    tools = [
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
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="Phi-3-mini-128k-directml-int4-awq-block-128-onnx",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=False,
    )
    for r in response:
        print(r)
    # if response:
    #     print(response)
    #     # content = response.choices[0].message.content
    #     # print(content)
    # else:
    #     print("Error:", response.status_code)


if __name__ == "__main__":
    begintime = time.time()
    function_chat()
    endtime = time.time()
    print("Time used:", endtime - begintime)
