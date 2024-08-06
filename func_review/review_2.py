from loguru import logger

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


def create_review_request(old_code, new_code, call_chain):
    prompt = """
Please review the following code changes, focusing on code quality and potential bugs.

The issues are categorized into three levels:
Critical Errors
Definition: Errors that cause system crashes, data loss, or threaten system security.
Examples: Null pointer exception causing the entire system to crash, critical data accidentally deleted, serious security vulnerabilities.
Major Errors
Definition: Errors that severely affect system functionality but do not cause a complete system crash.
Examples: Key functional module failure, user operations blocked or data inconsistency, inefficient algorithms leading to slow system response.
Minor Errors
Definition: Issues that affect some users' experience or minor functionality but the core functionality remains available.
Examples: Interface layout issues, non-critical function buttons not responding correctly.
Please describe the changes, potential issues, their levels, improvement suggestions, and analyze the associated impacts based on the function call chain, providing a final summary of the impacts in Chinese in the format of a JSON Object Array:

Old Code:
```python
{old_code}
```

New Code:
```python
{new_code}
```

Function Call graph, "->" indicates the call relationship:
```
{call_chain}
```
"""
    # 简化 prompt 的构建
    prompt = f"""请审查以下代码变更，关注点包括代码质量和潜在的bug.\n    
问题级别分为3类：
1. 致命错误（Critical）
定义：导致系统崩溃、数据丢失或系统安全受到威胁的错误。
示例：空指针异常导致整个系统瘫痪，重要数据被意外删除，存在严重安全漏洞。         
2. 严重错误（Major）
定义：严重影响了系统的功能，但系统未完全崩溃。
示例：关键功能模块失效，用户操作受阻或数据不一致，低效的算法导致系统响应缓慢。          
3. 一般错误（Minor）
定义：影响了部分用户的使用体验或功能的次要问题，但核心功能仍然可用。
示例：界面布局问题，非关键功能的按钮响应不正确。\n        
请以简短的语句描述可能的问题（issue)，问题的级别(level)、改进建议(suggest), 并根据函数调用链分析关联影响,给最终的关联影响生成中文总结(affected)，请以JSON Object Array的格式回答：\n
旧代码:
```python
{old_code}
```\n
新代码:
```python
{new_code}
```
函数调用链，"->"表示调用关系：\n
{call_chain}
"""
    try:
        response = client.chat.completions.create(
            model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
            n=1,
            stop=None,
            response_format={"type": "json_object"},
        )
        review = response.choices[0]
        return review
    except Exception as e:
        logger.error(f"创建审查请求时异常: {e}")
        return "审查请求过程中出现错误。"


def display_review_results(review):
    print("==== 代码审查结果 ====")
    print(review)
    print("========================")


if __name__ == "__main__":
    old_code = """
def add(a, b):
    return a + b
    """

    new_code = """
def add(a, b):
    result = a + b
    if result > 10:
        print("Result is too large")
    if result > 500:
      return result
    """
    call_chain = """
    getOrderTotalPrice（获取订单总价） -> getOrderCount
    getOrderCount -> add
    """
    review = create_review_request(old_code, new_code, call_chain)
    display_review_results(review)
