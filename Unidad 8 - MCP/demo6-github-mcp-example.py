from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools # type: ignore
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

async def main():
    mcp_api_key = os.getenv("GITHUB_MCP_API_KEY")

    async with streamablehttp_client("https://api.githubcopilot.com/mcp/", headers={"Authorization": f"Bearer {mcp_api_key}"}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            agent = create_agent("openai:gpt-4.1-nano", tools)
            response = await agent.ainvoke({"messages": "Create a new repository called 'my-new-web' and add a web page (HTML file) that contains information about Pokemons."})
            return response

if __name__ == "__main__":
    result = asyncio.run(main())
    print(result)