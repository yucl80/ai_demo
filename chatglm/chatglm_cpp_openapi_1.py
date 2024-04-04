from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"]="NOKEY" 
client = OpenAI(base_url="http://127.0.0.1:8000/v1/")


response = client.chat.completions.create(
    model="default-model",
    response_format={ "type": "json_object" },
    messages=[
    {"role": "system", "content": "As a senior business architect in the securities field.You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "What are the business domains and business sub-domains of securities companies? Please answer using json"}]
    )
print(response.choices[0].message.content)

