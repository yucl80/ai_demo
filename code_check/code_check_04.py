import openai

# 设置你的OpenAI API密钥
api_key = 'your_openai_api_key'

# 初始化OpenAI的API客户端
openai.api_key = api_key

def get_analysis_prompt(code_snippet, analysis_type, context="", call_chain="", examples=""):
    """
    生成用于代码分析的提示。

    :param code_snippet: 要分析的代码片段
    :param analysis_type: 分析类型（functionality, issues_and_suggestions, security, code_quality, refactoring, all）
    :param context: 额外的上下文信息
    :param call_chain: 函数调用链信息
    :param examples: 示例驱动的分析方法
    :return: 生成的提示字符串
    """
    prompts = {
        "functionality": f"请详细分析以下代码的功能并生成相应的文档：\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n示例：{examples}\n\n```python\n{code_snippet}\n```",
        "issues_and_suggestions": f"请详细分析以下代码可能存在的问题，并按严重性分级（高、中、低）列出具体的改进建议，例如潜在的bug、性能问题或代码冗余。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n示例：{examples}\n\n```python\n{code_snippet}\n```",
        "security": f"请详细分析以下代码可能存在的安全漏洞，并按严重性分级（高、中、低）列出具体的修复建议，例如潜在的SQL注入、跨站点脚本（XSS）漏洞或其他常见的安全问题。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n示例：{examples}\n\n```python\n{code_snippet}\n```",
        "code_quality": f"请详细评估以下代码的质量和可读性，并按严重性分级（高、中、低）列出改进建议，包括代码结构、命名约定和代码风格，以确保代码的一致性和可维护性。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n示例：{examples}\n\n```python\n{code_snippet}\n```",
        "refactoring": f"请详细分析以下代码并提供具体的重构建议，帮助优化代码的结构和性能，使其更加清晰和高效，并按严重性分级（高、中、低）列出建议。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n示例：{examples}\n\n```python\n{code_snippet}\n```",
        "all": f"请详细分析以下代码，包括功能理解、可能存在的问题或改进建议、安全漏洞、代码质量评估以及重构建议，并按严重性分级（高、中、低）列出。请尽可能详细，并提供实际案例。\n\n上下文信息：{context}\n\n函数调用链：{call_chain}\n\n示例：{examples}\n\n```python\n{code_snippet}\n```"
    }

    return prompts.get(analysis_type, prompts["all"])

def code_analysis(code_snippet, analysis_type="all", context="", call_chain="", examples=""):
    """
    根据指定的分析类型分析代码片段。

    :param code_snippet: 要分析的代码片段
    :param analysis_type: 分析类型（functionality, issues_and_suggestions, security, code_quality, refactoring, all）
    :param context: 额外的上下文信息
    :param call_chain: 函数调用链信息
    :param examples: 示例驱动的分析方法
    :return: 分析结果文本
    """
    prompt = get_analysis_prompt(code_snippet, analysis_type, context, call_chain, examples)

    try:
        # 发送请求给OpenAI的API
        response = openai.Completion.create(
            engine="text-davinci-003",  # 使用更强大的OpenAI模型
            prompt=prompt,
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

def multi_step_analysis(code_snippet, context="", call_chain="", examples=""):
    """
    使用多步骤分析对代码片段进行综合分析。

    :param code_snippet: 要分析的代码片段
    :param context: 额外的上下文信息
    :param call_chain: 函数调用链信息
    :param examples: 示例驱动的分析方法
    :return: 综合分析结果文本
    """
    steps = ["functionality", "issues_and_suggestions", "security", "code_quality", "refactoring"]
    results = {}

    for step in steps:
        results[step] = code_analysis(code_snippet, analysis_type=step, context=context, call_chain=call_chain, examples=examples)
    
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
        if "无严重错误" not in result:
            filtered_results[step] = result
    
    return filtered_results

# 主程序入口
if __name__ == "__main__":
    code_snippet = """
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)
    """

    context = "这是一个计算阶乘的递归函数。"
    call_chain = "factorial -> '*' operator -> factorial"
    examples = "示例1：计算5的阶乘。"

    print("=== Static Analysis Result ===")
    print(integrate_static_analysis_tools(code_snippet))
    print("\n" + "="*80 + "\n")

    print("=== Dynamic Analysis Result ===")
    print(dynamic_analysis(code_snippet))
    print("\n" + "="*80 + "\n")

    # 进行多步骤分析并打印结果
    analysis_results = multi_step_analysis(code_snippet, context=context, call_chain=call_chain, examples=examples)

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
