from openai import OpenAI
import time
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

model_id = "firefunction"

start_time = time.time()

messages = [
    {
        "role": "system",
        "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary",
    },
    {"role": "user", "content": "Extract Jason is 25 years old"},
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "UserDetail",
            "parameters": {
                "type": "object",
                "title": "UserDetail",
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "age": {"title": "Age", "type": "integer"},
                },
                "required": ["name", "age"],
            },
        },
    }
]

tool_choice = {"type": "function", "function": {"name": "UserDetail"}}

chat_completion = client.chat.completions.create(
    model=model_id,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
)
end_time = time.time()
print("Time taken: ", end_time - start_time)

print(chat_completion)


chat_completion = client.chat.completions.create(
    model=model_id,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"team_name": {"type": "string"}},
            "required": ["team_name"],
        },
    },
    temperature=0.7,
)

print(chat_completion)

chat_completion = client.chat.completions.create(
    model=model_id,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that outputs in JSON.",
        },
        {"role": "user", "content": "Who won the world series in 2020"},
    ],
    response_format={
        "type": "json_object",       
    },
    temperature=0.7,
)

print(chat_completion)