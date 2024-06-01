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

model_id = "functionary"

response = client.chat.completions.create(model=model_id,messages=messages,temperature=0.1,tool_choice="auto");


from pprint import pprint

messages = [    
    {'role': 'user', 'content': 'Return a list of named entities in the text.Text: The Golden State Warriors are an American professional basketball team based in San Francisco.Named entities:.'},
]
response = client.chat.completions.create(model=model_id,messages=messages,temperature=0.1,tool_choice="auto");
pprint(response)

messages = [    
    {'role': 'user', 'content': 'Translate the English text to Chinese.Text: Sometimes, I\'ve believed as many as six impossible things before breakfast.Translation:'},
]
response = client.chat.completions.create(model=model_id,messages=messages,temperature=0.1,tool_choice="auto");
pprint(response)

prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
Write a summary of the above text.
Summary:
"""

messages = [    
    {'role': 'user', 'content': prompt},
]
response = client.chat.completions.create(model=model_id,messages=messages,temperature=0.1,tool_choice="auto");
pprint(response)

prompt = """Answer the question using the context below.
Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
Question: What modern tool is used to make gazpacho?
Answer:
"""
messages = [    
    {'role': 'user', 'content': prompt},
]
response = client.chat.completions.create(model=model_id,messages=messages,temperature=0.1,tool_choice="auto");
pprint(response)

prompt = """There are 5 groups of students in the class. Each group has 4 students. How many students are there in the class?"""
messages = [    
    {'role': 'user', 'content': prompt},
]
response = client.chat.completions.create(model=model_id,messages=messages,temperature=0.1,tool_choice="auto");
pprint(response)

prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
Date:"""

messages = [    
    {'role': 'user', 'content': prompt},
]
response = client.chat.completions.create(model=model_id,messages=messages,temperature=0.1,tool_choice="auto");
pprint(response)