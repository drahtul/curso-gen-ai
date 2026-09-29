from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools # type: ignore
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

async def main():
    # La credencial de GitHub habilita un servidor MCP remoto con herramientas
    # capaces de producir efectos persistentes sobre repositorios.
    mcp_api_key = os.getenv("GITHUB_MCP_API_KEY")

    # Streamable HTTP conecta el agente con el servidor oficial de GitHub;
    # load_mcp_tools descubrira sus operaciones y contratos de entrada.
    async with streamablehttp_client("https://api.githubcopilot.com/mcp/", headers={"Authorization": f"Bearer {mcp_api_key}"}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            # El agente no solo genera texto: puede delegar acciones operativas
            # al modelo y a las tools descubiertas.
            agent = create_agent("openai:gpt-4.1-nano", tools)
            # Esta instruccion solicita crear recursos externos. En un entorno
            # real deben revisarse permisos, alcance del token y reversibilidad.
            response = await agent.ainvoke({"messages": "Create a new repository called 'my-new-web' and add a web page (HTML file) that contains information about Pokemons."})
            return response

if __name__ == "__main__":
    result = asyncio.run(main())
    print(result)