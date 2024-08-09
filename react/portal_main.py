from openai import OpenAI
import os
import re
import json
from portal_functions import (
    get_function_definitions,
    get_function_names,
    validate_function_call,
    get_functions,
)
from entity_vectors import search_entities, get_entity_info
from graph_query import execute_graph_query

client = OpenAI(base_url="http://localhost:8000/v1/", api_key="api_key")


def generate_context(query: str) -> str:
    relevant_entities = search_entities(query)
    context = "Relevant information:\n"
    for entity in relevant_entities:
        info = get_entity_info(entity)
        context += f"- {entity}: {json.dumps(info)}\n"
    return context


def llm(messages: list, functions: list) -> dict:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        function_call="auto",
        max_tokens=300,
    )
    return response.choices[0].message.content


def create_system_message(function_names: str, context: str) -> dict:
    examples = """
    Example 1:
    User query: What's the weather like today?
    Thought: The query is about weather, but the location is uncertain. I need to consider the following:
    1. The user might be asking about their current location.
    2. There might be a default location set.
    3. We might need to ask the user for a specific location.
    Action: ask_user_info
    Action Input: "To provide accurate weather information, could you please specify the location you're interested in?"
    User: San Francisco, CA
    Action: get_weather
    Action Input: {"location":"San Francisco, CA"}

    Example 2:
    User query: Calculate the result of x + y
    Thought: This is a calculation query, but the values of x and y are not provided. There's uncertainty in the input.
    Action: ask_user_info
    Action Input: "To perform the calculation, I need the values for x and y. Could you please provide them?"
    User: 10 + 20
    Action: calculate
    Action Input: {"expression":"10 + 20"}
  
    Example 3:
    User query: Tell me about the Golden Gate Bridge
    Thought: This is a general query about the Golden Gate Bridge. While I have some information in the context, it might be limited. I'll provide what I know and acknowledge any limitations.
    Action: ask_user_info
    Action Input: "Based on the available information, I can tell you about the Golden Gate Bridge, but please note that my knowledge might be incomplete."
    """

    system_content = f"""You are an intelligent assistant that needs to analyze user queries and decide the next action. 
    Use the following context information to enhance your understanding:
    {context}
    
    Here are some examples of how to respond to different types of queries, including handling uncertainty:
    {examples}
    
    Follow these steps:
    1. Analyze the query and identify any uncertainties or ambiguities.
    2. If there are uncertainties, explain them in your thought process and consider how to address them.
    3. Determine which predefined functions ({function_names}) are suitable for the query, or if you need to ask for more information.
    4. If no predefined function is suitable, determine if the user's intent is to query information. 
    5. If user's intent is to query information do Action : graph_query.
    Always explain your thought process before deciding on the actions."""

    return {"role": "system", "content": system_content}


def react_agent(user_input: str) -> str:
    # context = generate_context(user_input)
    context = "系统组件ses,hsbroker.ses, 核心交易\n系统组件sis,hsbroker.sis,综合交易\n"
    prompt = f"User query: {user_input}\n"
    prompt += "Please analyze the user query and decide the next action based on the instructions and context provided."
    messages = []
    messages.append(
        create_system_message(function_names=get_function_names(), context=context)
    )
    messages.append({"role": "user", "content": prompt})
    max_iterations = 5
    for i in range(max_iterations):
        response = llm(messages=messages, functions=get_function_definitions())
        messages.append({"role": "assistant", "content": response})
        print(f"Thought: {response}")  # For debugging
        thought_match = re.search(r"Thought: (.*)", response)
        action_match = re.search(r"Action: (.*)", response)
        action_input_match = re.search(r"Action Input: (.*)", response)

        if action_match and action_input_match:
            # thought = thought_match.group(1)
            action = action_match.group(1)
            action_input = action_input_match.group(1)
            print(action_input)
            print(f"action:{action}, action input: {action_input}")
            if action == "ask_user_info":
                # todo read string from console
                print(action_input)
                user_answer = input(":")
                messages.append({"role": "user", "content": user_answer +"\nPlease through and analyze the user query and decide the next action."})
                continue
            
            elif action == "default_response":
                print("default_response")
                return action_input
            
            elif action == "graph_query":
                try:
                   result = execute_graph_query(str)
                   return result
                except SyntaxError as e:
                    print(f"Syntax error in graph query: {e}")
                    messages.append({"role": "user", "content": f"\nThe previous graph query {e.query} had a syntax error: {e}. Please generate a corrected query."})
                    continue
                
            else:
                validate_result = validate_function_call(action, action_input)
                print(validate_result["result"])
                if validate_result["result"]:
                    function_args = json.loads(action_input)
                    result = get_functions()[action](**function_args)
                    result = {"func_name": action, "func_args": function_args}
                    return result
                else:
                    errors = validate_result["errors"]
                    messages.append(
                        {
                            "role": "user",
                            "content": f"\nThe generated function call for {action} with arguments {action_input} is invalid. \n {errors} Please try again.",
                        }
                    )
                    continue       
        else:
            return response

    return "I apologize, but I couldn't provide a definitive answer within the allowed number of iterations. The query might be too complex or ambiguous. Could you please rephrase or provide more specific information?"


# Test cases
test_queries = [
    "2号核心链路图",
    # "哪些系统使用了Redis？"
    # "2号核心链路包含哪些API"
    # "依赖核心交易的组件有哪些？",
    # "What's the weather like today?",
    # "Calculate x + y",
    # "Who won the Nobel Prize last year?",
    # "Tell me about the Golden Gate Bridge",
    # "哪个组件提供了API：ses.postOrder?"
]

for query in test_queries:
    print(f"\nUser query: {query}")
    result = react_agent(query)
    print(f"Final answer: {result}")
