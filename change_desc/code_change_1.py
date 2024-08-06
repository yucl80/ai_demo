from loguru import logger

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")
from groq import Groq
GROQ_KEY="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"

client = Groq(
    api_key=GROQ_KEY,
)


# from openai import OpenAI

# client = OpenAI(base_url=base_url, api_key=api_key)

prompt="""
As a code review expert, summarize the function changes enclosed in triple backticks:

1. Be extremely concise and factual.
2. Focus only on key modifications.
3. Use brief, technical language.
4. Omit all opinions, suggestions, and impact analysis.

Rules:
- Respond in Chinese.
- Maximum 500 characters.
- Prioritize brevity over detail.

Changes to summarize:
```code```
[Function changes will be inserted here]
```code```
"""


def send_llm_request(prompt):
    try:
        response = client.chat.completions.create(
            # model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            # model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
           model="llama3-8b-8192",
        #  model="llama-3.1-70b-versatile",
        #  model="gemma2-9b-it",
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
        )
        print(response)
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


def create_summary_request(change_list):
    prompt = f"""作为代码审查助手。我将提供给你函数的变更描述以。请你简要总结变更描述内容，帮助审查者更快更方便地理解文件中的变化。总结必须完全客观，不含意见或建议。请务必使用中文回复，不要超过10个字。
        函数及函数的变更描述如下：      
        ```
        {change_list}
        ```\n      
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
    t1 = time.time() - begin_time
    display_review_results(review1)

    old_code = """
    public class OrderDao{
        /*asdfsdfasdfsa */
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

    begin_time = time.time()
    review2 = create_change_request(lang, old_code, new_code)
    t2 = time.time() - begin_time
    display_review_results(review2)

    print("used_time:", t1 + t2)

    change_list = f"""{review1.message.content}\{review2.message.content} """

    # 创建单个API的调用链变更的汇总
    api_change_summary = create_summary_request(change_list)

    print(api_change_summary)

    # 创建所有变更点的汇总
    # all_change_list = ""   #500
    # all_summary = create_summary_request(all_change_list)

 