# pip install zhipuai 请先在终端进行安装

from zhipuai import ZhipuAI

client = ZhipuAI(api_key="your api key")

response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {
            "role": "user",
            "content": "你好"
        }
    ],
    top_p=0.7,
    temperature=0.95,
    max_tokens=1024,
    stream=True,
)
for trunk in response:
    print(trunk)
    
    