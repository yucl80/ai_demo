from loguru import logger

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


def create_review_request(lang,old_code, new_code, call_chain):

    prompt = f"""请审查以下代码变更，根据调用关系分析最终的影响，关注点包括代码变更点及最终影响.  
    Please act as a code reviewer, review the  change. I want you to give:
     1. Determine whether the file contains major logic changes. Generally speaking,
     2. A brief summary of the diff change, no more than 100 words. Do not include the results of the first step       
        请以简短的语句描述变更(change), 最终的关联影响(affected),请以JSON Object Array的格式回答:\n
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
                {
                    "role": "system",
                    "content": "你作为一个名资深软件程序开发专家,请严格按照以下要求进行代码评审:1.请务必用中文回复。2.请简洁准确的总结改动内容和最终影响。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.1,
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
    public class OrderService{
   private double getOrderAmount(Order order){
        return calcAmout(order);
    }
    }
    """

    new_code = """
     public class OrderService{
   private double getOrderAmount(Order order){
        double amout = calcAmout(order);
        return amout * getOrderDiscounts(amout);
    }   

    private double getOrderDiscounts(double totalAmount) {
        if (totalAmount > 1000) {
            return 0.85;
        } else if (totalAmount > 500) {
            return 0.8;
        } else {
            return 0.99;
        }

    }
    }
    """
    call_chain = """
    getOrderTotalPrice -> getOrderAmount    
    getOrderAmount -> getOrderDiscounts
    """
    lang = "java"
    review = create_review_request(lang,old_code, new_code, call_chain)
    display_review_results(review)
