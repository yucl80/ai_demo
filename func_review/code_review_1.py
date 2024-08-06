from loguru import logger

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)


def create_review_request(lang,old_code, new_code):
   
    # 简化 prompt 的构建
    prompt = f"""作为代码审查助手。我将提供给你变更前后的代码。请审查代码的差异,关注点包括代码的质量,潜在的问题及改进建议    
        问题级别分为3类:
        1. 致命错误(ritical)
        定义：导致系统崩溃、数据丢失或安全漏洞，内存泄漏，系统安全受到威胁的错误。          
        2. 严重错误(Major)
        定义：会导致功能错误、金额计算精度问题，不正确的行为或性能问题，但不至于系统崩溃。                
        3. 一般错误(Minor)
        定义：不会影响系统的正常运行，但会影响代码的可读性、可维护性或引发潜在的小问题。              
        请以简短的语句描述函数名(function),可能的问题(issue)，问题的级别(level),改进建议(suggest)。
        回复请以JSON Object Array格式:[{{"function":"函数名称","issue":"问题描述","level":"问题级别","suggest":"改进建议"}}]:\n       
        旧代码:
        ```{lang}
        {old_code}
        ```
        新代码:
        ```{lang}
        {new_code}
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
    public class OrderService{
    private double getOrderAmount(Order order){
        return calcAmout(order);
    }
    }
    """

    new_code = """
    public class OrderService{
        private double getOrderAmount(Order order){
            double totalAmount = calcAmout(order);
            return totalAmount * getOrderDiscounts(totalAmount);
            
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
   
    review = create_review_request("java",old_code, new_code)
    display_review_results(review)
    
   