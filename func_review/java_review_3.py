from loguru import logger

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


def create_review_request(lang, new_code):
    prompt = f"""请审查以下Java代码,关注点包括代码质量、潜在的问题\n    
        问题级别分为3类:
        1. 致命错误(Critical)
        定义：导致系统崩溃、数据丢失或安全漏洞，内存泄漏，系统安全受到威胁的错误。          
        2. 严重错误(Major)
        定义：会导致功能错误、金额计算精度问题，不正确的行为或性能问题，但不至于系统崩溃。                
        3. 一般错误(Minor)
        定义：不会影响系统的正常运行，但会影响代码的可读性、可维护性或引发潜在的小问题。              
        请以简短的语句描述函数名(function),可能的问题(issue)，问题的级别(level)
        回复请以JSON Object Array格式:[{{"function":"函数名称","issue":"问题描述","level":"问题级别","suggest":"改进建议"}}]:\n        
        代码:
        ```{lang}
        {new_code}
        ```
        """        
       
    try:
        response = client.chat.completions.create(
            # model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            messages=[            
                {
                    "role": "system",
                    "content": "你作为一个名资深软件工程师,请先完整理解代码的功能然后再严格按照以下要求进行代码评审:1.请务必用中文回复。2.请简洁准确的总结问题。",
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
  
    new_code = """
     public class OrderService{
   private double getOrderAmount(Order order){
        double totalAmount = calcTotalAmout(order);
        return totalAmount * getOrderDiscounts(totalAmount);
    }   

    private double getOrderDiscounts(double totalAmount) {
        if (totalAmount >= 1000) {
            return 0.85;
        } else if (totalAmount > 500) {
            return 0.8;
        } else {
            return 0.99;
        }

    }
    }
    """
   
    lang = "java"
    review = create_review_request(lang, new_code)
    display_review_results(review)
