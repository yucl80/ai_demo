# Create server parameters for stdio connection
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI

import asyncio


async def main():
    # Define the MCP server's SSE endpoint
    sse_url = "http://127.0.0.1:8080/sse"

    # Initialize the SSE client transport
    # transport = SseClientTransport(sse_url)

    # Create the MCP client with the SSE transport

    async with sse_client("http://127.0.0.1:8080/sse") as streams:
        async with ClientSession(streams[0], streams[1]) as session:
      
            # Initialize the connection
            await session.initialize()

            print("after init")

            response = await session.list_tools()
            tools = response.tools

            # Connect to the MCP server
            # await client.connect()

            print(tools)

            # Example: Call a tool named 'add' with parameters
            result = await session.call_tool('add', {'a': 5, 'b': 3})
            print(f"Result: {result}")

            # Keep the connection alive to listen for events
            # await asyncio.sleep(3600)  # Keep alive for 1 hour

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
   
    asyncio.run(main())

