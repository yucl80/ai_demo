
api_key = "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"
base_url = "http://127.0.0.1:8000/v1/"

from groq import Groq

client = Groq( api_key=api_key)

response = client.chat.completions.create(
  model="gemma-7b-it",
messages=[
    {
      "role": "system",
      "content": "You will be provided with a text, and your task is to extract the airport codes from it."
    },
    {
      "role": "user",
      "content": "I want to fly from Orlando to Boston"
    }
  ],
  temperature=0.5,
  max_tokens=64,
  top_p=1
)
from pprint import pprint

pprint(response.choices[0].message.content)