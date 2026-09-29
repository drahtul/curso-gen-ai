import asyncio
import json
from fastmcp import Client

async def main():
    # El cliente se conecta a la interfaz MCP del servidor; no importa sus
    # funciones Python directamente ni conoce la implementacion interna.
    client = Client("demo1-weather-mcp-server.py")
    async with client:
        # list_tools implementa descubrimiento dinamico de nombres,
        # descripciones y esquemas de argumentos disponibles.
        result = await client.list_tools()
        result_dicts = [tool.__dict__ for tool in result]
        print(json.dumps(result_dicts, indent=2))

        # call_tool envia nombre y argumentos estructurados y recibe un
        # resultado MCP, sin acoplarse al codigo del servidor.
        forecast = await client.call_tool("get_forecast", {"latitude": 40.7128, "longitude": -74.0060})
        print("\nForecast Result:")
        # Un resultado puede contener varios bloques; este ejemplo toma el
        # primer bloque textual para mostrarlo.
        print(forecast.content[0].text)
        
asyncio.run(main())