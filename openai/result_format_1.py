import instructor
from pydantic import BaseModel
from openai import OpenAI


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int


# Patch the OpenAI client
client = instructor.from_openai(OpenAI(api_key="nokey",base_url="http://192.168.1.104:8000/v1/"))

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="lmstudio-community/qwen/qwen2.5-14b-instruct-q4_k_m.gguf",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)
print(user_info)
# print(user_info.name)
#> John Doe
# print(user_info.age)
#> 30