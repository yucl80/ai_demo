from zhipuai import ZhipuAI
from jwt_token import get_api_key
from openai import OpenAI
import os

client = ZhipuAI(api_key=get_api_key()) # 请填写您自己的APIKey

os.environ["OPENAI_API_KEY"]="NOKEY"

#client = OpenAI(base_url="http://127.0.0.1:8000/v1/")

response = client.chat.completions.create(
    model="glm-4", # 填写需要调用的模型名称
    messages = [
        {
            "role": "user",
            "content": "你能帮我查询2024年1月1日从北京南站到上海的火车票吗？我需要硬卧"
        }
    ],
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_train_info",
                "description": "根据用户提供的信息，查询对应的车次",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "departure": {
                            "type": "string",
                            "description": "出发城市或车站",
                        },
                        "destination": {
                            "type": "string",
                            "description": "目的地城市或车站",
                        },
                        "date": {
                            "type": "string",
                            "description": "要查询的车次日期",
                        },
                    },
                    "required": ["departure", "destination", "date"],
                },
            }
        }
    ],
    tool_choice="auto",
)
print(response.choices[0].message)