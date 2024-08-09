api_key = "your-api-key"
# base_url = "http://192.168.32.129:8000/v1/"
base_url = "http://127.0.0.1:8000/v1/"

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


def code_analysis(code_snippet, analysis_type="all"):
    # 根据分析类型设置提示
    if analysis_type == "functionality":
        prompt = f"请详细分析以下代码的功能并生成相应的文档：\n\n```python\n{code_snippet}\n```"
    elif analysis_type == "issues_and_suggestions":
        prompt = f"请详细分析以下代码可能存在的问题,并提出具体的改进建议,例如潜在的bug、性能问题或代码冗余:\n\n```python\n{code_snippet}\n```"
    elif analysis_type == "security":
        prompt = f"请详细分析以下代码可能存在的安全漏洞,例如潜在的SQL注入、跨站点脚本(XSS)漏洞或其他常见的安全问题，并提供修复建议：\n\n```python\n{code_snippet}\n```"
    elif analysis_type == "code_quality":
        prompt = f"请详细评估以下代码的质量和可读性，包括代码结构、命名约定和代码风格，并提供改进建议，以确保代码的一致性和可维护性：\n\n```python\n{code_snippet}\n```"
    elif analysis_type == "refactoring":
        prompt = f"请详细分析以下代码并提供具体的重构建议，帮助优化代码的结构和性能，使其更加清晰和高效：\n\n```python\n{code_snippet}\n```"
    else:
        prompt = f"请详细分析以下代码，包括功能理解、可能存在的问题或改进建议、安全漏洞、代码质量评估以及重构建议：\n\n```python\n{code_snippet}\n```"

    # 发送请求给OpenAI的API
    response = client.chat.completions.create(
        # model="codegeex4",  # 使用更强大的OpenAI模型
        model="deepseek-coder-v2-lite",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,  # 增加最大生成的标记数
        temperature=0.2,  # 设置较低的温度以获得更确定的结果
        timeout=24000,
    )
    print(response)

    return response.choices[0].message.content


# 示例代码
code_snippet = """
import psycopg2

def execute_query(user_input):
    conn = psycopg2.connect("dbname=test user=postgres password=secret")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{user_input}'"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

def calculate_square(n):
    result = n * n
    return result
"""

# 进行功能理解和文档生成
functionality_result = code_analysis(code_snippet, analysis_type="functionality")
print("功能理解和文档生成:")
print(functionality_result)
print("\n")

# 进行问题识别和建议
issues_and_suggestions_result = code_analysis(
    code_snippet, analysis_type="issues_and_suggestions"
)
print("问题识别和建议:")
print(issues_and_suggestions_result)
print("\n")

# 进行安全性分析
security_analysis_result = code_analysis(code_snippet, analysis_type="security")
print("安全性分析:")
print(security_analysis_result)
print("\n")

# 进行代码质量评估
code_quality_result = code_analysis(code_snippet, analysis_type="code_quality")
print("代码质量评估:")
print(code_quality_result)
print("\n")

# 进行重构建议
refactoring_result = code_analysis(code_snippet, analysis_type="refactoring")
print("重构建议:")
print(refactoring_result)
print("\n")

# 综合分析
comprehensive_analysis_result = code_analysis(code_snippet, analysis_type="all")
print("综合分析:")
print(comprehensive_analysis_result)
print("\n")
