import asyncio
import json
from fastmcp import Client
from dotenv import load_dotenv
import os

# Las credenciales se separan del codigo y se leen del entorno, no del prompt
# ni del repositorio fuente.
load_dotenv()
API_KEY = os.getenv("POKEMON_MCP_API_KEY")

async def main():
    # Esta URL usa transporte HTTP remoto; a diferencia del demo local, el
    # cliente no inicia un archivo Python como servidor hijo.
    client = Client(
        "https://parliamentary-gray-snipe.fastmcp.app/mcp",
        auth=API_KEY,
    )

    async with client:
        # El cliente descubre el contrato publicado por el servidor remoto.
        result = await client.list_tools()
        result_dicts = [tool.__dict__ for tool in result]
        print(json.dumps(result_dicts, indent=2))

        # auth protege el servidor MCP y es independiente de la credencial del
        # proveedor del LLM. Los argumentos se envian de forma estructurada.
        pokemon_info = await client.call_tool("get_pokemon_info", {"name": "snorlax"})
        print("\nPokemon Info:")

        # Esta es la frontera entre un bloque textual MCP y datos JSON que el
        # programa puede recorrer y formatear localmente.
        data = json.loads(pokemon_info.content[0].text)

        print(json.dumps(data, indent=2))


asyncio.run(main())