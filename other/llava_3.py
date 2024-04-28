import base64
import openai
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


# Helper function to encode the image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# The path to your image
image_path = "your_image.jpg"

# The base64 string of the image
image_base64 = encode_image(image_path)

client = openai.OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="<FIREWORKS_API_KEY>",
)
# response = client.chat.completions.create(
#     model="accounts/fireworks/models/firellava-13b",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                 "type": "text",
#                 "text": "Can you describe this image?",
#                 }, 
#                 {
#                     "type": "image_url",
#                     "image_url":
#                         {
#                         "url": f "data:image/jpeg;base64,{image_base64}"
#                         },
#                 },],
#         }
#         ]
#     )

# print(response.choices[0].message.content)
