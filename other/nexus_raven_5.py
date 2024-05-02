#from nexusraven import Client
from openai import OpenAI
import json

# Example adapted from: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

client = OpenAI(base_url="http://localhost:8000/v1", api_key="INSERT YOUR KEY")
# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API


def get_current_weather(location, unit="fahrenheit"):
    return location


def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "Can you get me the current weather for Paris, get it for San Francisco, and get it for New York?"}]
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
        model="ravenv2",
        messages=messages,
        functions=tools,
       # tool_choice="auto",  # auto is default, but we'll be explicit
    )
    print(response)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        # extend conversation with assistant's reply
        messages.append(response_message)
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        return tool_calls


print(run_conversation())
