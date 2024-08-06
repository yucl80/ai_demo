from loguru import logger

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


def create_review_request(old_code, new_code):
    # 简化 prompt 的构建
    prompt = f"""请审查以下代码变更，关注点包括代码质量和潜在的bug.\n    
        问题级别分为3类：
        1. 致命错误（Critical）
        定义：导致系统崩溃、数据丢失或系统安全受到威胁的错误。
        示例：空指针异常导致整个系统瘫痪，重要数据被意外删除。         
        2. 严重错误（Major）
        定义：严重影响了系统的功能，但系统未完全崩溃。
        示例：关键功能模块失效，如支付系统无法处理交易。          
        3. 一般错误（Minor）
        定义：影响了部分用户的使用体验，但核心功能仍然可用。
        示例：界面布局问题，非关键功能的按钮响应不正确。\n      
        请以简短的语句描述可能的问题（issue)，问题的级别(level)和改进建议(suggest),请以JSON Object Array的格式回答：\n
        \n旧代码:\n```python\n{old_code}\n```\n\n新代码:\n```python\n{new_code}\n```"""
    try:
        response = client.chat.completions.create(
            model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
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
    if result > 100:
      return result
    """

    review = create_review_request(old_code, new_code)
    display_review_results(review)
