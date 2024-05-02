"""
This script is an example of using the OpenAI API to create various interactions with a ChatGLM3 model.
It includes functions to:

1. Conduct a basic chat session, asking about weather conditions in multiple cities.
2. Initiate a simple chat in Chinese, asking the model to tell a short story.
3. Retrieve and print embeddings for a given text input.

Each function demonstrates a different aspect of the API's capabilities, showcasing how to make requests
and handle responses.
"""

from yucl.utils import get_api_token
from openai import OpenAI
import os
import time

base_url = "http://127.0.0.1:8000/v1/"
# base_url = "https://open.bigmodel.cn/api/paas/v4/"
os.environ["OPENAI_API_KEY"] = get_api_token()

client = OpenAI(base_url=base_url)

SYSTEM_PROMPT_FOR_CHAT_MODEL = """
    You are an expert in composing functions. You are given a question and a set of possible functions. 
    Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
    If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
    also point it out. You should only return the function call in tool_calls sections.
    """

USER_PROMPT_FOR_CHAT_MODEL2 = """
    Questions:{user_prompt}\nHere is a list of functions in JSON format that you can invoke:\n{functions}. 
    Should you decide to return the function call(s),Put it in the format of tool_calls=[{{function:{{arguments:'{{params_name: params_value}}',name:function_name}}, type='function'}}]\n
    NO other text MUST be included. 
"""
USER_PROMPT_FOR_CHAT_MODEL = """
    Questions:{user_prompt}\nHere is a list of functions in JSON format that you can invoke:\n{functions}. 
    Should you decide to return the function call(s),Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n
    NO other text MUST be included. 
"""


def function_chat():

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
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_FOR_CHAT_MODEL,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FOR_CHAT_MODEL.format(
                user_prompt="What's the weather like in San Francisco and WuHan?",
                functions=tools,
            ),
        },
    ]

    response = client.chat.completions.create(
        model="openfunctions",
        messages=messages,
        # tools=tools,
        tool_choice="auto",
        stream=False,
    )
    if response:
        print(response)
        # content = response.choices[0].message.content
        # print(content)
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    begintime = time.time()
    function_chat()
    endtime = time.time()
    print("Time used:", endtime - begintime)
