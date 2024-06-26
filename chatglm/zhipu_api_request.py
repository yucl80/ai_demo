"""
This script is an example of using the Zhipu API to create various interactions with a ChatGLM3 model. It includes
functions to:

1. Conduct a basic chat session, asking about weather conditions in multiple cities.
2. Initiate a simple chat in Chinese, asking the model to tell a short story.
3. Retrieve and print embeddings for a given text input.
Each function demonstrates a different aspect of the API's capabilities,
showcasing how to make requests and handle responses.

Note: Make sure your Zhipu API key is set as an environment
variable formate as xxx.xxx (just for check, not need a real key).
"""

from zhipuai import ZhipuAI
from yucl.utils import get_api_key
import os

#base_url = "http://127.0.0.1:8000/v1/"
base_url = "https://open.bigmodel.cn/api/paas/v4/"

#base_url = "http://127.0.0.1:8000/v1/"
client = ZhipuAI(api_key=get_api_key(), base_url=base_url)

def function_chat():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
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

    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    if response:
        content = response.choices[0].message.content
        print(content)
    else:
        print("Error:", response.status_code)




if __name__ == "__main__":
  
    function_chat()
