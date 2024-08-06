from loguru import logger

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


def create_review_request(lang,old_code, new_code, call_chain):
   
    # 简化 prompt 的构建
    prompt = f"""请审查以下代码变更，关注点包括代码变更点及关联影响.         
        请以简短的语句描述变更（change）， 并根据函数调用链分析关联影响,给最终的关联影响生成中文总结(affected)，请以JSON Object Array的格式回答：\n
        旧代码:
        ```{lang}
        {old_code}
        ```\n
        新代码:
        ```{lang}
        {new_code}
        ```\n
        函数调用链，"->"表示调用关系：
        ```
        {call_chain}
        ```
        """
    try:
        response = client.chat.completions.create(
            model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            messages=[
                # {"role":"system","content":"As a senior business application development expert, your task is to conduct a code review."},
                {"role":"system","content":"作为一个名资深业务应用程序开发专家，你的任务是进行代码评审."},
                {"role": "user", "content": prompt}],
            max_tokens=1000,
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


def summarize_impact_with_llm(review):
    prompt = f"""请以简短的语句总结下面内容,不超过200字：    
        ```
        {review}       
        ```
        """
    try:
        response = client.chat.completions.create(
            model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            messages=[
                # {"role":"system","content":"As a senior business application development expert, your task is to conduct a code review."},
                # {"role":"system","content":"作为一个名资深业务应用程序开发专家，你的任务是进行代码评审."},
                {"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2,
            n=1,
            stop=None,
            response_format={"type": "json_object"},
        )
        review = response.choices[0]
        return review
    except Exception as e:
        logger.error(f"创建审查总结请求时异常: {e}")
        return "请求过程中出现错误。"
    

if __name__ == "__main__":
    old_code = """
def addOrder(a, b):
    return a + b
    """

    new_code = """
def addOrder(a, b,productId):
    result = a + b * getProductPriceDiscounts(productId)   
    if result > 500:
      return result
      
def getProductPriceDiscounts(productId)
    if productId == 'cpu':
       return 1000
    else:
       return 200

    """
    call_chain = """
    getOrderTotalPrice（获取订单总价） -> getOrderCount   
    getOrderCount -> add
    getOrderCount -> getProductPriceDiscounts
    """
    review = create_review_request(old_code, new_code,call_chain)
    display_review_results(review)
    
    summarize = summarize_impact_with_llm(review)
    
    display_review_results(summarize)