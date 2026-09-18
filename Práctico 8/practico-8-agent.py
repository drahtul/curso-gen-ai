import asyncio
import os
from contextlib import AsyncExitStack
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import load_mcp_tools  # type: ignore
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from typing_extensions import TypedDict

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
mcp_api_key = os.getenv("PERSONAL_MCP_API_KEY")

MCP_SERVER_URL = "https://frantic-silver-alpaca.fastmcp.app/mcp"

@tool
def calcular_total(precios: list[float]) -> float:
    """Calcula el costo total de una compra sumando una lista de precios.

    Args:
        precios: Lista con el precio de cada producto de la compra (ej: [4500, 800])
    """
    return round(sum(precios), 2)


@tool
def calcular_envio(monto_compra: float) -> dict:
    """Calcula el costo de envío según el monto total de la compra.

    Reglas: menos de $1,000 -> $120; entre $1,000 y $5,000 -> $60; más de $5,000 -> gratis.

    Args:
        monto_compra: Monto total de la compra (sin envío)
    """
    if monto_compra < 1000:
        envio = 120.0
    elif monto_compra <= 5000:
        envio = 60.0
    else:
        envio = 0.0
    return {
        "monto_compra": monto_compra,
        "costo_envio": envio,
        "total_con_envio": round(monto_compra + envio, 2),
    }


LOCAL_TOOLS = [calcular_total, calcular_envio]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class TechStoreAgent:
    def __init__(self):
        self.llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-mini")
        self.graph = None
        self.session = None
        self.tools = []
        self.prompts = []
        self.base_system_prompt = ""
        self.active_prompt_name = None
        self.active_prompt_text = None
        self.checkpointer = InMemorySaver()
        self._exit_stack = AsyncExitStack()

    async def initialize(self):
        """Conecta con el servidor MCP y descubre sus tools, prompt y resource."""
        read, write, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(MCP_SERVER_URL, 
                                  headers={"Authorization": f"Bearer {mcp_api_key}"},
                                  )
        )
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        mcp_tools = await load_mcp_tools(self.session)
        self.tools = mcp_tools + LOCAL_TOOLS

        print(f"Tools descubiertas en el servidor MCP ({len(mcp_tools)}):")
        for t in mcp_tools:
            print(f"  - [MCP]   {t.name}")
        print(f"Tools locales ({len(LOCAL_TOOLS)}):")
        for t in LOCAL_TOOLS:
            print(f"  - [LOCAL] {t.name}")

        self.prompts = (await self.session.list_prompts()).prompts
        print(f"Prompts disponibles en el servidor MCP ({len(self.prompts)}):")
        for p in self.prompts:
            print(f"  - {p.name}: {p.description}")

        politica = await self.session.read_resource("techstore://politica-garantias")
        politica_texto = politica.contents[0].text

        self.base_system_prompt = (
            "Eres el asistente de TechStore. Usa las herramientas disponibles para consultar "
            "productos, precios y stock; no inventes datos.\n"
            "Herramientas de cálculo: usa calcular_total para sumar precios y calcular_envio "
            "para obtener el costo de envío a partir del monto de la compra.\n\n"
            f"Referencia - {politica_texto}"
        )

    async def activar_prompt(self, name: str):
        """Obtiene un prompt del servidor MCP y lo agrega al system prompt."""
        prompt_def = next((p for p in self.prompts if p.name == name), None)
        if prompt_def is None:
            print(f"No existe el prompt '{name}'. Usa /prompts para ver los disponibles.\n")
            return

        arguments = {}
        for arg in prompt_def.arguments or []:
            valor = input(f"  {arg.name} ({arg.description or 'valor'}): ").strip()
            if valor or arg.required:
                arguments[arg.name] = valor

        prompt = await self.session.get_prompt(name, arguments)
        self.active_prompt_name = name
        self.active_prompt_text = "\n".join(
            m.content.text for m in prompt.messages if m.content.type == "text"
        )
        print(f"Prompt '{name}' activado.\n")

    def desactivar_prompt(self):
        if self.active_prompt_name is None:
            print("No hay ningún prompt activo.\n")
            return
        print(f"Prompt '{self.active_prompt_name}' desactivado.\n")
        self.active_prompt_name = None
        self.active_prompt_text = None

    def listar_prompts(self):
        for p in self.prompts:
            marca = " (activo)" if p.name == self.active_prompt_name else ""
            print(f"  - {p.name}{marca}: {p.description}")
        print()

    async def close(self):
        await self._exit_stack.aclose()

    def build_graph(self):
        """Construye el grafo ReAct: agent <-> tools."""
        llm_with_tools = self.llm.bind_tools(self.tools)

        async def agent_node(state: AgentState):
            system_prompt = self.base_system_prompt
            if self.active_prompt_text:
                system_prompt = f"{self.active_prompt_text}\n\n{system_prompt}"
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")

        self.graph = graph_builder.compile(checkpointer=self.checkpointer)

    async def stream_tool_responses(self, user_input: str, thread_id: str = "default"):
        """Ejecuta el agente mostrando qué tools usa (MCP o local) en cada paso."""
        local_names = {t.name for t in LOCAL_TOOLS}
        config = {"configurable": {"thread_id": thread_id}}

        async for step in self.graph.astream(
            {"messages": [HumanMessage(content=user_input)]}, config
        ):
            node_name = list(step.keys())[0]
            last_msg = step[node_name]["messages"][-1]

            if node_name == "agent" and last_msg.tool_calls:
                for call in last_msg.tool_calls:
                    origen = "LOCAL" if call["name"] in local_names else "MCP"
                    print(f"  -> [{origen}] {call['name']}({call['args']})")
            elif node_name == "tools":
                for msg in step[node_name]["messages"]:
                    print(f"  <- {msg.name}: {msg.content}")
            else:
                print(f"\nAsesor: {last_msg.content}\n")


async def main():
    agent = TechStoreAgent()

    try:
        await agent.initialize()
        agent.build_graph()

        print("\n" + "=" * 80)
        print("TechStore - Asistente de ventas (LangGraph + MCP)")
        print("=" * 80)
        print("Ejemplos:")
        print("  - ¿Cuánto cuesta el teclado mecánico Logitech G413?")
        print("  - Si compro un monitor de $4,500 y un mouse de $800, ¿cuánto pagaré incluyendo el envío?")
        print("  - ¿Cuál es el precio del Logitech G305 y cuánto pagaré si agrego el envío?")
        print('  - ¿Qué productos tienen stock disponible dentro de la categoría "Monitores"?')
        print("Comandos:")
        print("  /prompts         lista los prompts del servidor MCP")
        print("  /usar <nombre>   activa un prompt (ej: /usar asesor_de_ventas)")
        print("  /quitar          desactiva el prompt activo")
        print("  exit             salir\n")

        while True:
            etiqueta = f"Tú [{agent.active_prompt_name}]: " if agent.active_prompt_name else "Tú: "
            user_query = input(etiqueta).strip()
            if user_query.lower() == "exit":
                print("¡Hasta luego!")
                break
            if not user_query:
                continue

            if user_query == "/prompts":
                agent.listar_prompts()
            elif user_query.startswith("/usar"):
                nombre = user_query.removeprefix("/usar").strip()
                if nombre:
                    await agent.activar_prompt(nombre)
                else:
                    print("Indica el nombre del prompt: /usar <nombre>\n")
            elif user_query == "/quitar":
                agent.desactivar_prompt()
            else:
                await agent.stream_tool_responses(user_query)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
