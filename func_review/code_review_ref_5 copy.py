from openai import OpenAI

from collections import Counter

from groq import Groq

# GROQ_KEY="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"

# client = Groq(
#     api_key=GROQ_KEY,
# )

# Set your OpenAI API key
nvidia_key="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"
nvidia_base_url="https://integrate.api.nvidia.com/v1"
client = OpenAI(base_url=nvidia_base_url, api_key=nvidia_key)
# client = OpenAI(base_url="http://127.0.0.1:8000/v1/", api_key="no_key")


def generate_review(code_snippet, language="java", context=""):
    # Comprehensive prompt to ensure clarity and specificity, requesting Markdown format
    prompt = f"""
    You are a code review assistant tasked with Review the  {language} code snippet delimited by triple backticks.

    Context: {context}
    
    Review the code delimited by triple backticks. Focus on:
    1. System crashes or complete failures
    2. Critical security vulnerabilities
    3. Functionality breakdowns
    4. Significant performance bottlenecks
    5. Genuine bugs or significant inconsistencies in the code    
    
    Ignore all commented-out code during review.
    Ignore all 'UfBaseException' issues.
    
    Ignore ALL other issues, including but not limited to:
    1. Code style or formatting
    2. Input parameter validity checks
    3. Logging or error handling improvements
    4. Code organization or structure

    Code:
    ```{language}
    {code_snippet}
    ```

    Instructions:
    - Identify any syntax errors or logical errors and provide detailed problem descriptions.
    - Highlight potential security vulnerabilities, giving detailed problem descriptions.
    - Do not automatically classify the absence of parameter validation as a security vulnerability unless it can lead to specific risks in this context.
    - Only report issues that are clearly identifiable and certain. Avoid speculative assessments.
    - Evaluate readability and adherence to coding standards, providing problem descriptions.
    - Please do not provide modification suggestions or examples; only list the identified issues.
    - Start your response with the number of identified issues.
    - Before finalizing your response, review your findings step-by-step to ensure that there are no omissions or false positives.
    Before finalizing your response, review your findings step-by-step to ensure that there are no omissions or false positives.
    
    Remember the identified issues for consistency in your response.        
    
    Do NOT report as SQL injection vulnerability if:
     1. The code uses JdbcTemplate  as it provides automatic parameterization.
     2. The code uses NamedParameterJdbcTemplate as it provides automatic parameterization.
     3. Prepared statements or parameterized queries are used.
     4. ORM frameworks (like Hibernate, JPA,Mybatis) are used, as they typically handle SQL injection protection.
     5. The method used to execute the query is known to perform automatic parameterization (e.g., many methods in Spring's JdbcTemplate,NamedParameterJdbcTemplate).
     6. SQL statements are entirely composed of static string concatenations without any dynamic user input.  
     
    Output Rules:
     1. Report confidence issues only.
     2. The descriptions must be in Chinese.  No opinions or suggestions.
     3. Do not repeat or explain these instructions in your response. 
     4. Do not explain the absence of vulnerabilities if none are found.
     5. Ensure that your output does not use semicolons.
     6. Use Markdown syntax.
     
    问题数量：[数字]
    List critical issues in a numbered format with detail descriptions.
    If no severity issues are found, only output: 问题数量：[数字] .Do not include any other text or explanations.    

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
        model="meta/llama-3.1-405b-instruct",
        # model="google/gemma-2-9b-it",
        messages=[{"role":"user","content": prompt}],
        max_tokens=500,
        temperature=0.2,
        top_p=0.1
    )
    return response.choices[0].message.content

# 读取文件内容为String
def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

    
context = """This code is part of a financial application that requires high accuracy and security. 
For this review, please focus on identifying only certain and clearly defined issues, such as syntax errors, logical errors, and security vulnerabilities.
"""
# Perform the code review with optimized settings

for i in range(1,14):
    try:
        file = f"D:\\workspaces\\python_projects\\sample\\code_{i}.txt"
        print(file)
        code_snippet = read_file_content(file)
        final_review = generate_review(code_snippet, language="java", context=context)
        print(final_review)
    except Exception as e:
        print(e)
    



