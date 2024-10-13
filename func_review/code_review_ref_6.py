from openai import OpenAI
from collections import Counter

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
    prompt = f"""
You are a code review assistant tasked with reviewing the `{language}` code snippet delimited by triple backticks.

Context:
```{context}```

Review the code concerning:
1. System crashes or complete failures
2. Critical security vulnerabilities
3. Functionality breakdowns
4. Significant performance bottlenecks
5. Genuine bugs or significant inconsistencies

During the review:
* Ignore all commented-out code
* Ignore 'UfBaseException' issues
* Ignore all other issues, including code style, input parameter checks, logging, error handling improvements, and code organization

The code snippet to review is:
```{language}
{code_snippet}
```

Report as a security issue only if:
* There is no use of `JdbcTemplate` or `NamedParameterJdbcTemplate` with automatic parameterization
* Prepared statements or parameterized queries are not used
* SQL injection is possible without ORM frameworks providing protection
* The query execution method doesn't automatically parameterize
* SQL strings are dynamically composed with user input

Before finalizing your report, review your findings step-by-step for completeness and accuracy. Remember identified issues for consistent reporting.

Output Guidelines:
* Report only confidence issues
* Use Chinese for all descriptions
* No opinions or suggestions
* Do not report absence of vulnerabilities
* Avoid semicolons
* Use Markdown for formatting

List critical issues in a numbered format with detailed descriptions.
      

    """
    
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
        model="lmstudio-community/Qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
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
    



