import openai

# 设置你的OpenAI API密钥
api_key = 'your_openai_api_key'

# 初始化OpenAI的API客户端
base_url = "http://127.0.0.1:8000/v1/"

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)

def get_analysis_prompt(code_snippet, analysis_type, context="", call_chain=""):
    """
    生成用于代码分析的提示。

    :param code_snippet: 要分析的代码片段
    :param analysis_type: 分析类型（functionality, issues_and_suggestions, security, code_quality, refactoring, all）
    :param context: 额外的上下文信息
    :param call_chain: 函数调用链信息
    :return: 生成的提示字符串
    """
    prompts = {
        "functionality": f"请详细分析以下代码的功能并生成相应的文档：\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n```python\n{code_snippet}\n```",
        "issues_and_suggestions": f"请详细分析以下代码可能存在的问题，并按严重性分级（高、中、低）列出具体的改进建议，例如潜在的bug、性能问题或代码冗余。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n```python\n{code_snippet}\n```",
        "security": f"请详细分析以下代码可能存在的安全漏洞，并按严重性分级（高、中、低）列出具体的修复建议，例如潜在的SQL注入、跨站点脚本（XSS）漏洞或其他常见的安全问题。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n```python\n{code_snippet}\n```",
        "code_quality": f"请详细评估以下代码的质量和可读性，并按严重性分级（高、中、低）列出改进建议，包括代码结构、命名约定和代码风格，以确保代码的一致性和可维护性。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n```python\n{code_snippet}\n```",
        "refactoring": f"请详细分析以下代码并提供具体的重构建议，帮助优化代码的结构和性能，使其更加清晰和高效，并按严重性分级（高、中、低）列出建议。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n```python\n{code_snippet}\n```",
        "all": f"请详细分析以下代码，包括功能理解、可能存在的问题或改进建议、安全漏洞、代码质量评估以及重构建议，并按严重性分级（高、中、低）列出。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n```python\n{code_snippet}\n```"
    }

    return prompts.get(analysis_type, prompts["all"])

def code_analysis(code_snippet, analysis_type="all", context="", call_chain=""):
    """
    根据指定的分析类型分析代码片段。

    :param code_snippet: 要分析的代码片段
    :param analysis_type: 分析类型（functionality, issues_and_suggestions, security, code_quality, refactoring, all）
    :param context: 额外的上下文信息
    :param call_chain: 函数调用链信息
    :return: 分析结果文本
    """
    prompt = get_analysis_prompt(code_snippet, analysis_type, context, call_chain)

    try:
        # 发送请求给OpenAI的API
        response = client.chat.completions.create(
            model="text-davinci-003",  # 使用更强大的OpenAI模型
            messages=[{"role":"user","content":prompt}],
            max_tokens=2000,  # 增加最大生成的标记数
            temperature=0.2,  # 设置较低的温度以获得更确定的结果
        )

        # 解析OpenAI API的响应
        if response and 'choices' in response:
            # 获取生成的文本
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
        else:
            return "未能生成分析结果。"
    except openai.OpenAIError as e:
        return f"API调用失败：{str(e)}"

def multi_step_analysis(code_snippet, context="", call_chain=""):
    """
    使用多步骤分析对代码片段进行综合分析。

    :param code_snippet: 要分析的代码片段
    :param context: 额外的上下文信息
    :param call_chain: 函数调用链信息
    :return: 综合分析结果文本
    """
    steps = ["functionality", "issues_and_suggestions", "security", "code_quality", "refactoring"]
    results = {}

    for step in steps:
        results[step] = code_analysis(code_snippet, analysis_type=step, context=context, call_chain=call_chain)
    
    return results

def integrate_static_analysis_tools(code_snippet):
    """
    使用静态分析工具对代码片段进行初步检查。

    :param code_snippet: 要分析的代码片段
    :return: 静态分析工具的输出
    """
    # 示例：调用Pylint进行静态分析（实际使用时需要安装和配置Pylint）
    # from pylint import epylint as lint
    # pylint_stdout, pylint_stderr = lint.py_run(code_snippet, return_std=True)
    # return pylint_stdout.getvalue()

    # 这里用假数据模拟静态分析工具的输出
    return "Pylint: 代码质量评分 8.5/10，无严重错误。"

def dynamic_analysis(code_snippet):
    """
    使用动态分析工具对代码片段进行检查。

    :param code_snippet: 要分析的代码片段
    :return: 动态分析工具的输出
    """
    # 示例：调用动态分析工具（如Coverage）进行分析（实际使用时需要安装和配置工具）
    # import coverage
    # cov = coverage.Coverage()
    # cov.start()
    # exec(code_snippet)
    # cov.stop()
    # cov.save()
    # return cov.report()

    # 这里用假数据模拟动态分析工具的输出
    return "Coverage: 代码覆盖率90%，所有功能均通过测试。"

def post_process_results(analysis_results):
    """
    对分析结果进行后处理，过滤误报。

    :param analysis_results: 多步骤分析结果
    :return: 过滤后的分析结果
    """
    filtered_results = {}
    
    for step, result in analysis_results.items():
        # 示例：简单过滤某些关键词
        if "无误报" not in result and "false positive" not in result:
            filtered_results[step] = result
    
    return filtered_results

# 示例代码
code_snippet = """
import psycopg2

def execute_query(user_input):
    # 连接到数据库
    conn = psycopg2.connect("dbname=test user=postgres password=secret")
    cursor = conn.cursor()
    
    # 执行SQL查询
    query = f"SELECT * FROM users WHERE username = '{user_input}'"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # 关闭连接
    conn.close()
    return rows

def calculate_square(n):
    # 计算平方
    result = n * n
    return result
"""

context = """
该项目是一个用户管理系统，主要功能包括用户查询和一些数学计算。数据库使用PostgreSQL。
"""

call_chain = """
1. 主程序调用 execute_query(user_input)
2. execute_query(user_input) 调用 psycopg2.connect(...)
3. psycopg2.connect(...) 返回连接对象 conn
4. execute_query(user_input) 使用 conn 创建 cursor
5. execute_query(user_input) 执行 cursor.execute(query)
6. execute_query(user_input) 获取结果 rows 并返回
7. 主程序调用 calculate_square(n)
8. calculate_square(n) 计算并返回结果 result
"""

# 初步静态分析
static_analysis_result = integrate_static_analysis_tools(code_snippet)
print("=== Static Analysis Result ===")
print(static_analysis_result)
print("\n" + "="*80 + "\n")

# 动态分析
dynamic_analysis_result = dynamic_analysis(code_snippet)
print("=== Dynamic Analysis Result ===")
print(dynamic_analysis_result)
print("\n" + "="*80 + "\n")

# 进行多步骤分析并打印结果
analysis_results = multi_step_analysis(code_snippet, context)

# 后处理结果，过滤误报
filtered_analysis_results = post_process_results(analysis_results)

for step, result in filtered_analysis_results.items():
    print(f"=== {step.replace('_', ' ').capitalize()} ===")
    print(result)
    print("\n" + "="*80 + "\n")

# 结合所有步骤的结果进行总结
summary = "综合分析结果：\n"
for step, result in filtered_analysis_results.items():
    summary += f"{step.replace('_', ' ').capitalize()}:\n{result}\n\n"

print(summary)
