from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-xxx")
response = client.chat.completions.create(
    model="llava",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://img0.baidu.com/it/u=3795936406,3301328154&fm=253&fmt=auto&app=138&f=JPEG?w=600&h=400"
                    },
                },
                {"type": "text", "text": "What does the image say"},
            ],
        }
    ],
)
print(response)