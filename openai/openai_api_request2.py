from openai import OpenAI
import time

base_url = "http://127.0.0.1:8000/v1/"
# base_url = "https://open.bigmodel.cn/api/paas/v1/"
client = OpenAI(api_key="APIKEY", base_url=base_url)

model_id = "chatglm"


def function_chat():
    messages = [
        {
            "role": "user",
            "content": "中国平安的股票价格是多少?",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "获取股票的当前价格",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stock_symbol": {
                            "type": "string",
                            "description": "股票代码"
                        }
                    },
                    "required": [
                        "stock_symbol"
                    ]

                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(response)
    if response:
        content = response.choices[0].message.content
        print(content)
    else:
        print("Error:", response.status_code)


def simple_chat(use_stream=False):
    messages = [
        {
            "role": "system",
            "content": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's "
            "instructions carefully. Respond using markdown.",
        },
        {
            "role": "user",
            "content": "你好，请你用生动的话语给我讲一个小故事吧, 故事要不少于50字",
        },
    ]
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        stream=use_stream,
        max_tokens=1024,
        temperature=0.8,
        presence_penalty=1.1,
        top_p=0.8,
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content, end="")
        else:
            content = response.choices[0].message.content
            print(content)
    else:
        print("Error:", response.status_code)


def embedding():
    response = client.embeddings.create(
        model="bge-large-zh-v1.5",
        input=["你好，给我讲一个故事，大概100字"],
    )
    embeddings = response.data[0].embedding
    print("嵌入完成，维度：", len(embeddings))


if __name__ == "__main__":

    #    simple_chat(use_stream=True)

    begin_time = time.time()
    simple_chat(use_stream=True)
    # embedding()
    # function_chat()
    end_time = time.time()
    print("总耗时：", end_time - begin_time)
