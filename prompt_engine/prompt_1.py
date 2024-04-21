from openai import OpenAI
import json


base_url = "http://127.0.0.1:8000/v1/"

client = OpenAI(api_key="ENTER_YOUR_API_KEY_HERE", base_url=base_url)

messages = [
  
    {'role': 'system', 'content': 'Answer in a consistent style..'},
    {'role': 'user', 'content': 'Teach me about patience.'},
     {'role': 'assistant', 'content': 'The river that carves the deepest valley flows from a modest spring; the grandest symphony originates from a single note; the most intricate tapestry begins with a solitary thread.'},
    {'role': 'user', 'content': 'Teach me about the ocean.'},
]


response = client.chat.completions.create(model="chatglm3",messages=messages,temperature=0.1,tool_choice="auto");


from pprint import pprint

pprint(response)