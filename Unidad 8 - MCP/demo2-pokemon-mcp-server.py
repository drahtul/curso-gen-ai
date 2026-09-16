from typing import Any
import httpx

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("poke-search")

@mcp.tool()
async def get_pokemon_info(name: str) -> Any:
    """Get information about a specific Pokemon.

    Args:
        name: The name of the Pokémon to search for
    """

    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        filtered_data = {
            "name": data.get("name"),
            "height": data.get("height"),
            "weight": data.get("weight"),
            "types": [t["type"]["name"] for t in data.get("types", [])],
            "abilities": [a["ability"]["name"] for a in data.get("abilities", [])],
            "base_experience": data.get("base_experience")
        }
        result = filtered_data
    return result


def main():
    print("MCP server is running. Press Ctrl+C to stop.")
    mcp.run()
    

if __name__ == "__main__":
    main()