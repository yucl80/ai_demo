from loguru import logger

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)

e_prompt="""Act as a Code Reviewer Assistant. I will provide you with the code before and after the changes.
And I want you to briefly summarize the content of the diff to helper reviewers understand what happened in this file faster and more convienently.
Your summary must be totaly objective and contains no opinions or suggestions.
Please reply in Chinese using a summary format, and keep it under 30 characters.\n"""

def send_llm_request(prompt):
    try:
        response = client.chat.completions.create(
            # model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            messages=[
                # {
                #     "role": "system",
                #     "content": "你作为一个名资深软件程序开发专家,你的任务是进行代码变更评审。",
                # },
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
        logger.error(f"创建请求时异常: {e}")
        return "请求过程中出现错误。"



def create_change_request(lang, old_code, new_code):
    prompt = f"""作为代码审查助手。我将提供给你变更前后的代码。请你简要总结差异内容，帮助审查者更快更方便地理解文件中的变化。总结必须完全客观，不含意见或建议。请务必使用中文回复，不要超过20个字。
    代码如下：
        旧代码:
        ```{lang}
        {old_code}
        ```\n
        新代码:
        ```{lang}
        {new_code}
        ```      
        """
    return send_llm_request(prompt)


def create_summary_request(change_list, call_chain):
    prompt = f"""作为代码审查助手。我将提供给你函数的变更描述以。请你简要总结差异内容及分析最终影响，帮助审查者更快更方便地理解文件中的变化及关联影响。总结必须完全客观，不含意见或建议。请务必使用中文回复，不要超过20个字。
        函数及函数的变更描述如下：      
        ```
        {change_list}
        ```\n
        函数调用关系（"->"表示调用关系）如下:
        ```
        {call_chain}
        ```      
        """
    return send_llm_request(prompt)


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

   
    }
    """

    lang = "java"
    import time

    begin_time = time.time()
    review1 = create_change_request(lang, old_code, new_code)
    end_time = time.time()
    print("used_time:", end_time - begin_time)
    display_review_results(review1)
    
    
    old_code = """
    public class OrderDao{
    private double queryOrder(Order order){
        return jdbcTemplate.query(order);
    }
    }
    """

    new_code = """
    public class OrderDao{
    private double queryOrder(Order order){
        List<Order> orderList = jdbcTemplate.query(order);
        orderList = orderList.stream().filter(o -> o.getOrderStatus() == OrderStatus.Paid ).collect(Collectors.toList());
        return order;
    }    
    }
    """
    review2 = create_change_request(lang, old_code, new_code)   
    display_review_results(review2)
    
    call_chain = """
    getOrderTotalPrice（获取订单总价） -> getOrderAmount   
    getOrderTotalPrice（获取订单总价） -> queryAllOrder
    queryAllOrder -> OrderDao
    """
    
    change_list = f"""getOrderAmount:{review1.message.content}\nOrderDao:{review2.message.content} """
    
    s = create_summary_request(change_list, call_chain)
    
    print(s)
