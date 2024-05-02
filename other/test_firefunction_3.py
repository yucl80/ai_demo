import os
import requests
import re
import openai


content = None
file_path = '/home/test/src/state_of_the_union.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

clean_content = content.replace("\n\n", "\n")[:8192]

client = openai.OpenAI(
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "YOUR_FW_API_KEY",
)
model_name = "firefunction"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_answer_with_sources",
            "description": "Answer questions from the user while quoting sources.",
            "parameters": {
                "type": "object",
                "properties": {
                  "answer": {
                      "type": "string",
                      "description": "Answer to the question that was asked."
                  },
                  "sources": {
                      "type": "array",
                      "items": {
                          "type": "string",
                          "description": "Source used to answer the question"
                      }
                  }
                },
                "required": ["answer", "sources"],
            }
        }
    }
]
tool_choice = {"type": "function", "function": {"name":"get_answer_with_sources"}}

messages = [
    {"role": "system", "content": f"You are a helpful assistant who is given document with following content: {clean_content}."
     															"Please reply in succinct manner and be truthful in the reply."},
    {"role": "user", "content": "What did the president say about Ketanji Brown Jackson?"}
]

chat_completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    temperature=0.01,
    response_format={ "type": "json_object" },
)

print(chat_completion.choices[0].message.model_dump_json(indent=4))

agent_response = chat_completion.choices[0].message

messages.append(
    {
        "role": agent_response.role,
        "content": "",
        "tool_calls": [
            tool_call.model_dump()
            for tool_call in agent_response.tool_calls
        ]
    }
)

messages.append(
    {
        "role": "user",
        "content": "What did he say about her predecessor?"
    }
)
next_chat_completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    temperature=0.1
)


print(next_chat_completion.choices[0].message.model_dump_json(indent=4))

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_answer_with_countries",
            "description": "Answer questions from the user while quoting sources.",
            "parameters": {
                "type": "object",
                "properties": {
                  "answer": {
                      "type": "string",
                      "description": "Answer to the question that was asked."
                  },
                  "countries": {
                      "type": "array",
                      "items": {
                          "type": "string",
                      },
                      "description": "countries mentioned in the sources"
                  }
                },
                "required": ["answer", "countries"],
            }
        }
    }
]
tool_choice = {"type": "function", "function": {"name":"get_answer_with_countries"}}

agent_response = next_chat_completion.choices[0].message

messages.append(
    {
        "role": agent_response.role,
        "content": "",
        "tool_calls": [
            tool_call.model_dump()
            for tool_call in agent_response.tool_calls
        ]
    }
)

messages.append(
    {
        "role": "user",
        "content": "What did he say about human traffickers?"
    }
)

chat_completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    temperature=0.1
)

print(chat_completion.choices[0].message.model_dump_json(indent=4))

