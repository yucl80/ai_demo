from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1",api_key="nokey")

response = client.chat.completions.create(
  model="Phi-3-mini-128k-directml-int4-awq-block-128-onnx",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)
print(response)
print(response.choices[0].message.content)