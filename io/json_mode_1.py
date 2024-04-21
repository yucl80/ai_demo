import openai
from pydantic import BaseModel, Field
import time

client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="Your_API_Key",
)

class Result(BaseModel):
    winner: str
    
print(Result.model_json_schema())

begin_time = time.time()

chat_completion = client.chat.completions.create(
    model="firefunction",
    response_format={"type": "json_object", "schema": Result.model_json_schema()},
    messages=[
        {
            "role": "user",
            "content": "Who won the US presidential election in 2012? Reply using JSON format. like: {\"winner\": \"Trump\"}",
        },
    ],
)
end_time = time.time()

print(f"Time taken: {end_time - begin_time} seconds")

print(repr(chat_completion.choices[0].message.content))