


import os

from groq import Groq

GROQ_KEY="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"

client = Groq(
    api_key=GROQ_KEY,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models\n Output rules: 1. use chinese ",
        }
    ],
    model="gemma2-9b-it",
)

print(chat_completion.choices[0].message.content)