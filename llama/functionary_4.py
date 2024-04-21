from openai import OpenAI
import time
client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

start_time = time.time()
response = client.chat.completions.create(
    model="functionary",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant with access to functions. Use them if required.",
        },
        {"role": "user", "content": "Hi, can you tell me the current stock price of AAPL?"},
       # {"role": "user", "content": "please use function to get what's the weather like in Hanoi?"},
    ],
    tools=[  # For functionary-7b-v2 we use "tools"; for functionary-7b-v1.4 we use "functions" = [{"name": "get_current_weather", "description":..., "parameters": ....}]
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
    ],
    tool_choice="auto"
)


# messages = [
#     {"role": "user", "content": "what's the weather like in Hanoi?"}
# ]


end_time = time.time()
print("Time taken: ", end_time - start_time)

print(response)
