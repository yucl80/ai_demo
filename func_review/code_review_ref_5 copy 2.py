from openai import OpenAI
from collections import Counter
from textwrap import dedent
from groq import Groq
GROQ_KEY="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"
client = Groq(
    api_key=GROQ_KEY,
)

deepseek_key="sk-3768d4b14e654ada8b1dc5f5007a0b16"
deepseek_base_url="https://api.deepseek.com/v1"
client = OpenAI(base_url=deepseek_base_url, api_key=deepseek_key)


# Set your OpenAI API key
nvidia_key="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"
nvidia_base_url="https://integrate.api.nvidia.com/v1"
# client = OpenAI(base_url=nvidia_base_url, api_key=nvidia_key)
client = OpenAI(base_url= "http://127.0.0.1:8000/v1/", api_key="api_key")


def generate_review(code_snippet, language="java", context=""):
    # Comprehensive prompt to ensure clarity and specificity, requesting Markdown format
    prompt =   dedent(
        f"""\
Please review the code below and identify any syntax or logical errors, suggest
ways to refactor and improve code quality, enhance performance, address security
concerns, and align with best practices. Provide specific examples for each area
and limit your recommendations to three per category.

Use the following response format, keeping the section headings as-is, and provide
your feedback. Use bullet points for each response. The provided examples are for
illustration purposes only and should not be repeated.

**Syntax and logical errors (example)**:
- Incorrect indentation on line 12
- Missing closing parenthesis on line 23

**Code refactoring and quality (example)**:
- Replace multiple if-else statements with a switch case for readability
- Extract repetitive code into separate functions

**Performance optimization (example)**:
- Use a more efficient sorting algorithm to reduce time complexity
- Cache results of expensive operations for reuse

**Security vulnerabilities (example)**:
- Sanitize user input to prevent SQL injection attacks
- Use prepared statements for database queries

**Best practices (example)**:
- Add meaningful comments and documentation to explain the code
- Follow consistent naming conventions for variables and functions

Code:
```{language}
{code_snippet}
```
  
 Your review:"""
    )  
    
    # **Example Output**:     
    # **问题数量: 2**
    # 1. **逻辑错误**:
    # - 函数不能处理负数，输入负数时会报错.
    # 2. **安全漏洞**:
    # - 存在SQL注入漏洞.

    # Single LLM call for comprehensive review
    response = client.chat.completions.create(
         # model="meta/llama3-70b-instruct",
        # model="meta/llama-3.1-405b-instruct",
        # model="meta/llama-3.1-8b-instruct",
        # model="google/gemma-2-9b-it",      
        # model="llama3-8b-8192",
        # model= "llama-3.1-8b-instant",
        # model= "mixtral-8x7b-32768",
        # model="deepseek-coder",
        model="deepseek-chat",
        messages=[{"role":"user","content": prompt}],
        max_tokens=500,
        temperature=0.2,
        top_p=0.1,
        timeout=100000
    )
    return response.choices[0].message.content

# 读取文件内容为String
def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

import re

def remove_comments(code):
    code = re.sub(r'//[^\n]*','',code)
    code = re.sub(r'/\*.*?\*/','',code,flags=re.DOTALL)
    return code
    
context = """This code is part of a financial application that requires high accuracy and security. 
For this review, please focus on identifying only certain and clearly defined issues, such as syntax errors, logical errors, and security vulnerabilities.
"""
# Perform the code review with optimized settings

for i in range(1,13):
    try:
        file = f"D:\\workspaces\\python_projects\\sample\\sql_{i}.txt"
        print(file)
        code_snippet = read_file_content(file)
        code_snippet = remove_comments(code_snippet)
        final_review = generate_review(code_snippet, language="java", context=context)
        print(final_review)
    except Exception as e:
        print(e)
    



