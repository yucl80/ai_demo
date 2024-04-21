import asyncio
import openai

client = openai.AsyncOpenAI(
    base_url = "http://127.0.0.1:8000/v1",
    api_key="sTn70521Te5NWDpDJGoyKHpU5jJxix2PtvRwH8bfjzfiKSUW",
)

async def main():
    stream = await client.completions.create(
        model="mixtral-8x7b-instruct",
        prompt="请编一个关于人类登录月球的故事",
        stream=True,
        max_tokens=200,
    )
    async for chunk in stream:
        print(chunk.choices[0].text, end="")

asyncio.run(main())