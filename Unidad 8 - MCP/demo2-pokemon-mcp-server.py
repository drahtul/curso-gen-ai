from typing import Any
import httpx

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("poke-search")

@mcp.tool()
async def get_pokemon_info(name: str) -> Any:
    # MCP publica una operacion especializada que un cliente o agente puede
    # solicitar bajo demanda, en lugar de exponer toda la API de PokéAPI.
    """Get information about a specific Pokemon.

    Args:
        name: The name of the Pokémon to search for
    """

    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
    async with httpx.AsyncClient() as client:
        # MCP es la interfaz de la herramienta; PokéAPI es el servicio externo
        # que realmente proporciona los datos.
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        # Filtrar la respuesta reduce ruido, tokens y superficie de datos que
        # llegara al modelo generativo.
        filtered_data = {
            "name": data.get("name"),
            "height": data.get("height"),
            "weight": data.get("weight"),
            "types": [t["type"]["name"] for t in data.get("types", [])],
            "abilities": [a["ability"]["name"] for a in data.get("abilities", [])],
            "base_experience": data.get("base_experience")
        }
        result = filtered_data
    # Se devuelven datos estructurados; el cliente decide si los serializa como
    # texto JSON para mostrarlos o para pasarlos a otro componente.
    return result


def main():
    print("MCP server is running. Press Ctrl+C to stop.")
    mcp.run()
    

if __name__ == "__main__":
    main()