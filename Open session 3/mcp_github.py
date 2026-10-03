"""Cliente MCP hacia el servidor oficial de GitHub (servidor de terceros).

Las tools se obtienen por protocolo MCP (streamable HTTP) con
langchain_mcp_adapters, igual que en la Unidad 8. No hay ninguna función
nuestra que "simule" a GitHub: lo que el subagente ejecuta es lo que el
servidor MCP expone.
"""
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore

load_dotenv()

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

# El servidor de GitHub expone decenas de tools (crear repos, PRs, borrar
# ramas...). El subagente solo necesita consultar, así que se le pasa un
# subconjunto de lectura: menos tokens para gpt-4.1-nano y ningún riesgo de
# que modifique algo en la cuenta del usuario.
TOOLS_PERMITIDAS = {
    "search_repositories",
    "get_file_contents",
    "search_code",
    "search_issues",
    "list_issues",
    "search_users",
}


async def cargar_tools_github() -> list:
    """Se conecta al servidor MCP de GitHub y devuelve sus tools de lectura."""
    client = MultiServerMCPClient(
        {
            "github": {
                "url": GITHUB_MCP_URL,
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('GITHUB_MCP_API_KEY')}",
                    # Modo solo-lectura del lado del servidor, como segunda barrera.
                    "X-MCP-Readonly": "true",
                },
            }
        }
    )
    todas = await client.get_tools()
    tools = [t for t in todas if t.name in TOOLS_PERMITIDAS]

    print(f"[mcp] GitHub MCP conectado: {len(todas)} tools expuestas, "
          f"{len(tools)} habilitadas -> {', '.join(t.name for t in tools)}")
    return tools
