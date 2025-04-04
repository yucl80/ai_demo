from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

import asyncio
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="lmstudio-community/qwen/qwen2.5-14b-instruct-q4_k_m.gguf",base_url="http://192.168.239.1:8000/v1", api_key="nokey")

async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["D:/workspaces/python_projects/mcp-server/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8000
                "url": "http://127.0.0.1:8080/sse",
                "transport": "sse",
            }
        }
    ) as client:
        tools = client.get_tools()
        print(tools)
        agent = create_react_agent(model, tools)
        weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"},debug=True)  

        print(weather_response)
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        print(math_response)
        

if __name__ == "__main__":
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())