import openai
from pydantic import BaseModel, Field

client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="Your_API_Key",
)

class Result(BaseModel):
    winner: str
    
print(Result.model_json_schema())

chat_completion = client.chat.completions.create(
    model="mixtral-8x7b-instruct",
    response_format={"type": "json_object", "schema": Result.model_json_schema()},
    messages=[
        {
            "role": "user",
            "content": "Who won the US presidential election in 2012? Reply just in one JSON.",
        },
    ],
)

print(repr(chat_completion.choices[0].message.content))