import asyncio
import json
from fastmcp import Client
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("POKEMON_MCP_API_KEY")

async def main():
    client = Client(
        "https://parliamentary-gray-snipe.fastmcp.app/mcp",
        auth=API_KEY,
    )

    async with client:
        
        result = await client.list_tools()
        result_dicts = [tool.__dict__ for tool in result]
        print(json.dumps(result_dicts, indent=2))

        pokemon_info = await client.call_tool("get_pokemon_info", {"name": "snorlax"})
        print("\nPokemon Info:")

        data = json.loads(pokemon_info.content[0].text)

        print(json.dumps(data, indent=2))


asyncio.run(main())