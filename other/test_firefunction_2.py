import openai
import json

client = openai.OpenAI(
 #   base_url = "https://api.fireworks.ai/inference/v1",
    
    base_url = "http://127.0.0.1:8000/v1/",
    api_key = "wrTGJ0yeD9BTheuMyeLl6kqic2rG3GtoL3jkaAWH7dmX1JrI"
)

messages = [
    {"role": "system", "content": f"You are a helpful assistant with access to functions." 
     															"Use them if required."},
    {"role": "user", "content": "What are Nike's net income in 2022?"}
]

tools = [
    {
        "type": "function",
        "function": {
            # name of the function 
            "name": "get_financial_data",
            # a good, detailed description for what the function is supposed to do
            "description": "Get financial data for a company given the metric and year.",
            # a well defined json schema: https://json-schema.org/learn/getting-started-step-by-step#define
            "parameters": {
                # for OpenAI compatibility, we always declare a top level object for the parameters of the function
                "type": "object",
                # the properties for the object would be any arguments you want to provide to the function
                "properties": {
                    "metric": {
                        # JSON Schema supports string, number, integer, object, array, boolean and null
                        # for more information, please check out https://json-schema.org/understanding-json-schema/reference/type
                        "type": "string",
                        # You can restrict the space of possible values in an JSON Schema
                        # you can check out https://json-schema.org/understanding-json-schema/reference/enum for more examples on how enum works
                        "enum": ["net_income", "revenue", "ebdita"],
                    },
                    "financial_year": {
                        "type": "integer", 
                        # If the model does not understand how it is supposed to fill the field, a good description goes a long way 
                        "description": "Year for which we want to get financial data."
                    },
                    "company": {
                        "type": "string",
                        "description": "Name of the company for which we want to get financial data."
                    }
                },
                # You can specify which of the properties from above are required
                # for more info on `required` field, please check https://json-schema.org/understanding-json-schema/reference/object#required
                "required": ["metric", "financial_year", "company"],
            },
        },
    }
]

chat_completion = client.chat.completions.create(
    model="firefunction-v1",
    messages=messages,
    tools=tools,
    temperature=0.1
)
print(chat_completion.choices[0].message.model_dump_json(indent=4))

def get_financial_data(metric: str, financial_year: int, company: str):
    print(f"{metric=} {financial_year=} {company=}")
    if metric == "net_income" and financial_year == 2022 and company == "Nike":
        return {"net_income": 6_046_000_000}
    else:
        raise NotImplementedError()


# function_call = chat_completion.choices[0].message.tool_calls[0].function
# tool_response = locals()[function_call.name](**json.loads(function_call.arguments))
# print(tool_response)

# agent_response = chat_completion.choices[0].message

# # Append the response from the agent
# messages.append(
#     {
#         "role": agent_response.role, 
#         "content": "",
#         "tool_calls": [
#             tool_call.model_dump()
#             for tool_call in chat_completion.choices[0].message.tool_calls
#         ]
#     }
# )

# # Append the response from the tool 
# messages.append(
#     {
#         "role": "assistant",
#         "content": json.dumps(tool_response)
#     }
# )

# next_chat_completion = client.chat.completions.create(
#     model="firefunction-v1",
#     messages=messages,
#     tools=tools,
#     temperature=0.1
# )

# print(next_chat_completion.choices[0].message.content)