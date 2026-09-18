"""Agente inteligente TechStore con LangGraph y Model Context Protocol (MCP).

Combina herramientas locales de cálculo comercial con herramientas descubiertas
dinámicamente desde el servidor MCP de TechStore.
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import load_mcp_tools  # type: ignore
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing_extensions import TypedDict

# Carga de variables de entorno desde la raíz del workspace

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# ==========================================
# Herramientas Locales del Agente
# ==========================================

@tool
def calcular_total(precios: list[float]) -> float:
    """Calcula el costo total de una compra a partir de una lista de precios individuales de productos.

    Args:
        precios: Lista de números con los precios de cada producto a comprar (ej: [4500.0, 800.0]).
    """
    total = sum(precios)
    return round(total, 2)


@tool
def calcular_costo_envio(monto_compra: float) -> dict[str, Any]:
    """Calcula el costo de envío correspondiente según el monto total de la compra.

    Reglas de negocio:
    - Compras menores a $1,000 → Costo de envío: $120.
    - Compras entre $1,000 y $5,000 (inclusive) → Costo de envío: $60.
    - Compras mayores a $5,000 → Costo de envío: $0 (Envío gratuito).

    Args:
        monto_compra: Monto total de los productos comprados (sin incluir el envío).
    """
    if monto_compra < 1000.0:
        costo_envio = 120.0
        tramo = "Compras menores a $1,000 ($120 de envío)"
    elif 1000.0 <= monto_compra <= 5000.0:
        costo_envio = 60.0
        tramo = "Compras entre $1,000 y $5,000 ($60 de envío)"
    else:
        costo_envio = 0.0
        tramo = "Compras mayores a $5,000 (Envío gratuito)"

    total_con_envio = monto_compra + costo_envio

    return {
        "monto_compra": monto_compra,
        "costo_envio": costo_envio,
        "total_con_envio": total_con_envio,
        "regla_aplicada": tramo,
    }


# ==========================================
# Definición del Estado del Grafo
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ==========================================
# Agente TechStore con LangGraph + MCP
# ==========================================

class TechStoreAgent:
    def __init__(self, model_name: str = "gpt-4.1-nano"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.0,
        )
        self.local_tools = [calcular_total, calcular_costo_envio]
        self.mcp_tools = []
        self.all_tools = []
        self.llm_with_tools = None
        self.graph = None
        self.session = None
        self.checkpointer = InMemorySaver()
        self._exit_stack = AsyncExitStack()

        # Prompt del sistema acorde al Asesor de Ventas
        self.system_prompt = SystemMessage(
            content=(
                "Eres un Asesor de Ventas profesional, cordial y comercial de la tienda TechStore.\n"
                "Tienes acceso a dos tipos de herramientas:\n"
                "1. Herramientas MCP (remotas del servidor TechStore): para buscar productos, consultar precios exactos, verificar stock y listar categorías.\n"
                "2. Herramientas Locales: para calcular el costo total de listas de compras (`calcular_total`) y calcular costos de envío (`calcular_costo_envio`).\n\n"
                "Instrucciones clave:\n"
                "- Si el usuario te pregunta por precios, stock o disponibilidad de productos en catálogo, consulta siempre las herramientas MCP.\n"
                "- Si el usuario te pide calcular un total o el envío de una compra, usa las herramientas locales apropiadas.\n"
                "- Si se combinan ambas necesidades (ej: consultar el precio de un producto y sumarle el envío), combina la herramienta MCP para obtener el precio y la herramienta local para calcular el envío.\n"
                "- Responde siempre de forma clara, detallada y cordial."
            )
        )

    async def initialize(self):
        """Inicia la conexión stdio con el servidor MCP y descubre las herramientas dinámicamente."""
        server_path = os.path.join(os.path.dirname(__file__), "techstore_server.py")

        print(f"[*] Conectando al servidor MCP en: {server_path}")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_path],
            env=dict(os.environ),
        )

        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        # Carga dinámica de herramientas expuestas por el servidor MCP
        self.mcp_tools = await load_mcp_tools(self.session)
        self.all_tools = [*self.mcp_tools, *self.local_tools]

        print(f"[+] Herramientas descubiertas desde Servidor MCP ({len(self.mcp_tools)}):")
        for t in self.mcp_tools:
            print(f"    - [MCP]   {t.name}: {t.description.splitlines()[0] if t.description else ''}")

        print(f"[+] Herramientas locales del agente ({len(self.local_tools)}):")
        for t in self.local_tools:
            print(f"    - [LOCAL] {t.name}: {t.description.splitlines()[0] if t.description else ''}")

    async def close(self):
        """Cierra la sesión MCP."""
        await self._exit_stack.aclose()

    def build_graph(self):
        """Construye el grafo de LangGraph con el ciclo LLM <-> Herramientas."""
        self.llm_with_tools = self.llm.bind_tools(self.all_tools)

        def agent_node(state: AgentState):
            messages = state["messages"]
            # Si el primer mensaje no es de sistema, lo incluimos
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [self.system_prompt] + list(messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        tool_node = ToolNode(tools=self.all_tools)

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", END)

        self.graph = graph_builder.compile(checkpointer=self.checkpointer)
        print("[+] Grafo de LangGraph compilado exitosamente.\n")

    async def run_query(self, user_query: str, thread_id: str = "default") -> str:
        """Ejecuta una consulta sobre el grafo y muestra la traza de herramientas ejecutadas."""
        print("=" * 80)
        print(f"Usuario: {user_query}")
        print("-" * 80)

        config = {"configurable": {"thread_id": thread_id}}
        final_answer = ""

        async for step in self.graph.astream(
            {"messages": [HumanMessage(content=user_query)]},
            config,
        ):
            node_name = list(step.keys())[0]
            state = step[node_name]
            last_msg = state["messages"][-1]

            if node_name == "agent":
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_name = tc.get("name")
                        tipo = "[LOCAL]" if tool_name in ["calcular_total", "calcular_costo_envio"] else "[MCP]"
                        print(f"-> Agente solicita ejecutar {tipo} '{tool_name}' con args: {tc.get('args')}")
                else:
                    final_answer = last_msg.content
            elif node_name == "tools":
                for msg in state["messages"]:
                    print(f"<- Resultado de Tool '{getattr(msg, 'name', 'tool')}': {msg.content}")

        print("-" * 80)
        print(f"Respuesta Asesor TechStore:\n{final_answer}\n")
        return final_answer


# ==========================================
# Ejecución Principal y Ejemplos de Prueba
# ==========================================

async def run_practical_examples(agent: TechStoreAgent):
    """Ejecuta los 4 ejemplos solicitados en la letra del práctico."""
    print("\n" + "#" * 80)
    print("EJECUCIÓN DE LOS 4 ESCENARIOS DE PRUEBA DE LA LETRA")
    print("#" * 80 + "\n")

    ejemplos = [
        (
            "Ejemplo 1 (Solo MCP)",
            "¿Cuánto cuesta el teclado mecánico Logitech G413?",
            "thread-ejemplo-1",
        ),
        (
            "Ejemplo 2 (Solo Local)",
            "Si compro un monitor de $4,500 y un mouse de $800, ¿cuánto pagaré incluyendo el envío?",
            "thread-ejemplo-2",
        ),
        (
            "Ejemplo 3 (Combinación MCP + Local)",
            "¿Cuál es el precio del Logitech G305 y cuánto pagaré si agrego el envío?",
            "thread-ejemplo-3",
        ),
        (
            "Ejemplo 4 (Combinación de herramientas MCP)",
            "¿Qué productos tienen stock disponible dentro de la categoría 'Monitores'?",
            "thread-ejemplo-4",
        ),
    ]

    for titulo, consulta, thread_id in ejemplos:
        print(f"[{titulo}]")
        await agent.run_query(consulta, thread_id=thread_id)
        await asyncio.sleep(0.5)


async def interactive_mode(agent: TechStoreAgent):
    """Modo chat interactivo con el asesor de TechStore."""
    print("=" * 80)
    print("MODO INTERACTIVO - Chat con Asesor TechStore")
    print("Escribe tu consulta o 'salir'/'exit' para terminar.")
    print("=" * 80 + "\n")

    thread_id = "interactive-session"
    while True:
        try:
            user_input = input("Tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Hasta luego!")
            break

        await agent.run_query(user_input, thread_id=thread_id)


async def main():
    agent = TechStoreAgent()

    try:
        await agent.initialize()
        agent.build_graph()

        # Ejecutar los 4 ejemplos pedidos en la letra
        await run_practical_examples(agent)

        # Si se desea continuar en modo interactivo
        print("¿Deseas probar consultas adicionales en modo interactivo? (s/n)")
        try:
            resp = input().strip().lower()
            if resp in ["s", "si", "y", "yes"]:
                await interactive_mode(agent)
        except (EOFError, KeyboardInterrupt):
            pass

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
