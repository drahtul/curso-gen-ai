from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

github_mcp_api_key = os.getenv("GITHUB_MCP_API_KEY")
pokemon_mcp_api_key = os.getenv("POKEMON_MCP_API_KEY")

client = MultiServerMCPClient(
    {
        # Cada nombre logico identifica el origen de las herramientas. El
        # transporte y la credencial se configuran de forma independiente.
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "transport": "streamable_http",
            "headers": {"Authorization": f"Bearer {github_mcp_api_key}"},
        },
        "pokemon": {
            "url": "https://parliamentary-gray-snipe.fastmcp.app/mcp",
            "transport": "streamable_http",
            "headers": {"Authorization": f"Bearer {pokemon_mcp_api_key}"},
        },
    }
)

checkpointer = InMemorySaver()
# La memoria del hilo es comun al agente, aunque las herramientas provengan de
# servidores MCP y dominios distintos.

async def main():
    # get_tools une las herramientas descubiertas en ambos servidores. El LLM
    # recibe una sola coleccion, por lo que las descripciones deben ser claras
    # para evitar ambiguedad o colisiones de nombres.
    tools = await client.get_tools()
    agent = create_agent("openai:gpt-4.1-nano", tools, checkpointer=checkpointer)

    # thread_id conserva el contexto conversacional; no mezcla permisos ni
    # credenciales entre GitHub y Pokemon.
    config = {"configurable": {"thread_id": "conversation-1"}}

    print("Chat con el agente MCP (GitHub + Pokemon).")
    print("Escribe tu mensaje y presiona Enter.")
    print("Comandos: 'salir', 'exit' o 'quit' para terminar.\n")

    while True:
        try:
            user_input = input("Tu > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego!")
            break

        # El LLM puede elegir el dominio adecuado segun la intencion, pero cada
        # tool sigue ejecutandose en su servidor MCP de origen.
        response = await agent.ainvoke(
            {"messages": user_input},
            config,
        )
        print(f"\nAgente > {response['messages'][-1].content}\n")

if __name__ == "__main__":
    asyncio.run(main())
