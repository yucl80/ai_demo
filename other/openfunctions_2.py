from openai import OpenAI
import json


client = OpenAI(api_key="YOUR_API_KEY", base_url="http://127.0.0.1:8000/v1")


def get_prompt(user_query: str, functions: list = []) -> str:
    """
    Generates a conversation prompt based on the user's query and a list of functions.

    Parameters:
    - user_query (str): The user's query.
    - functions (list): A list of functions to include in the prompt.

    Returns:
    - str: The formatted conversation prompt.
    """
    system = "You are an AI programming assistant, utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    if len(functions) == 0:
        return f"{system}\n### Instruction: <<question>> {user_query}\n### Response: "
    functions_string = json.dumps(functions)
    return f"{system}\n### Instruction: <<function>>{functions_string}\n<<question>>{user_query}\n### Response: "


def get_gorilla_response(
    prompt='Call me an Uber ride type "Plus" in Berkeley at zipcode 94704 in 10 minutes',
    model="gorilla-openfunctions-v0",
    functions=[],
):

    try:
        completion = client.chat.completions.create(
            model="openfunctions",
            temperature=0.0,
            messages=[{"role": "user", "content": get_prompt(prompt, functions)}],
            # functions=functions,
        )
        return completion.choices[0]
    except Exception as e:
        print(e, model, prompt)


query = "What's the weather like in the two cities of Boston and San Francisco?"
functions = [
    {
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
]
response = get_gorilla_response(query, functions=functions)
print(response)
