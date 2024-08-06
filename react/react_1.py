import time
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1/", api_key= "your-api-key")

def send_llm_request(messages):
    try:       
        response = client.chat.completions.create(
            # model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
            n=1,
            stop=None,           
        )
        result = response.choices[0].message.content
        return result
    except Exception as e:
        print(e)      
        return "请求过程中出现错误。"
    
prompt="""
You are an AI assistant capable of reasoning and acting. Follow these steps for each task:

1. Thought: Analyze the task and break it down into steps. Consider what information or actions are needed.

2. Action: Determine the next action to take. This could be:
   - Search: Look up information (specify what to search for)
   - Calculate: Perform a calculation (specify the calculation)
   - Ask: Request more information from the user (specify the question)

3. Observation: Based on the action, provide the result or information obtained.

4. Repeat steps 1-3 until you have enough information to complete the task.

5. Answer: Provide the final answer or solution to the task.

Always show your work by including each Thought, Action, Observation, and Answer step. Be thorough in your reasoning and explain your thought process clearly.
"""    

messages=[
    {"role":"system","content":prompt },
    {"role":"user","content":"哪些组件依赖了组件A?"}
]

result=send_llm_request(messages)
print(result)