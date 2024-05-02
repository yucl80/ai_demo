def get_car_price(car_name: str):
    """this function is used to get the price of the car given the name
    :param car_name: name of the car to get the price
    """
    car_price = {
        "rhino": {"price": "$20000"},
        "elephant": {"price": "$25000"} 
    }
    for key in car_price:
        if key in car_name.lower():
            return {"price": car_price[key]}
    return {"price": "unknown"}

functions = [
    {
        "name": "get_car_price",
        "description": "this function is used to get the price of the car given the name",
        "parameters": {
            "type": "object",
            "properties": {
                "car_name": {
                    "type": "string",
                    "description": "name of the car to get the price"
                }
            },
            "required": ["car_name"]
        }
    }
]

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

messages = [
    {
        "role": "user",
        "content": "What is the price of the car named 'Rhino'?"
    }
]

assistant_msg = client.chat.completions.create(
    model="functionary",
    messages=messages,
    functions=functions,
    temperature=0.0,
).choices[0].message

print(assistant_msg)

import json

def execute_function_call(message):   
    if message.function_call.name == "get_car_price":
        car_name = json.loads(message.function_call.arguments)["car_name"]
        results = get_car_price(car_name)
    else:
        results = f"Error: function {message.function_call.name} does not exist"
    return results

if assistant_msg.tool_calls is not None:
    results = execute_function_call(assistant_msg)
    print(results)
    # messages.append({"role": "assistant", "name": assistant_msg.function_call.name, "content": assistant_msg.function_call.arguments})
    # messages.append({"role": "function", "name": assistant_msg.function_call.name, "content": str(results)})
    messages.append(
    {
        "role": assistant_msg.role,
        "content": "",
        "tool_calls": [
            tool_call.model_dump()
            for tool_call in assistant_msg.tool_calls
        ],
    }
)
    messages.append({"role": assistant_msg.role, "content": json.dumps(results)})
    
    output_msg = client.chat.completions.create(
        model="chatglm3",
        messages=messages,
        functions=functions,
        temperature=0.0,
    ).choices[0].message
    
    print(output_msg)


