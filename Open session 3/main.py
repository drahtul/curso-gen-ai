"""CLI del sistema multiagente.

Uso:
    python main.py                      # usuario 'danilo', conversación nueva
    python main.py --user ana           # otro usuario (otra memoria de largo plazo)
    python main.py --thread charla-1    # retomar un thread (solo dentro del mismo proceso)

Comandos dentro del chat: 'memorias' (lista lo que se recuerda del usuario),
'olvidar' (borra la memoria de largo plazo del usuario), 'salir'.
"""
import argparse
import asyncio
import uuid

from langchain_core.messages import HumanMessage

from agente import construir_grafo
from mcp_github import cargar_tools_github
from memoria import borrar_memorias, listar_memorias


async def crear_sistema():
    # La conexión MCP es opcional: el grafo se construye aunque GitHub no esté
    # disponible, lo que permite seguir usando RAG, memoria y APIs propias.
    try:
        tools_mcp = await cargar_tools_github()
    except Exception as exc:  # sin MCP el resto del sistema sigue funcionando
        print(f"[mcp] No se pudo conectar al servidor MCP de GitHub: {exc}")
        tools_mcp = []
    return construir_grafo(tools_mcp)


async def preguntar(graph, consulta: str, user_id: str, thread_id: str) -> str:
    """Ejecuta un turno e imprime el resumen de auditoría."""
    # El thread_id identifica la memoria conversacional de corto plazo; el
    # user_id, en cambio, identifica la memoria semántica persistente.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    print("\n--- traza ---")
    estado = await graph.ainvoke(
        {"messages": [HumanMessage(content=consulta)], "user_id": user_id},
        config,
    )

    informes = estado.get("informes", [])
    print("\n--- auditoría ---")
    print(f"  subagentes: {[i['subagente'] for i in informes] or 'ninguno (respondió sin delegar)'}")
    print(f"  tools:      {[t for i in informes for t in i['tools']] or 'ninguna'}")
    print(f"  memorias recuperadas: {estado.get('memorias') or 'ninguna'}")
    print(f"  memorias guardadas:   {estado.get('memorias_guardadas') or 'ninguna'}")
    return estado["messages"][-1].content


async def main() -> None:
    # La CLI separa la configuración de identidad/sesión del contenido de cada
    # consulta, para poder demostrar threads y usuarios diferentes.
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="danilo")
    parser.add_argument("--thread", default=None)
    args = parser.parse_args()
    thread_id = args.thread or f"sesion-{uuid.uuid4().hex[:8]}"

    graph = await crear_sistema()

    print("=" * 80)
    print("Asistente multiagente - películas, libros, recetas, clima, países y GitHub")
    print(f"usuario: {args.user} | thread: {thread_id}")
    print("Comandos: 'memorias', 'olvidar', 'salir'")
    print("=" * 80)

    while True:
        # Leer en un hilo evita bloquear el event loop mientras input espera al
        # usuario; así las operaciones asíncronas conservan su modelo de ejecución.
        try:
            consulta = (await asyncio.to_thread(input, "\nConsulta: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not consulta:
            continue
        if consulta.lower() in {"salir", "exit", "quit"}:
            print("¡Hasta luego!")
            break
        if consulta.lower() == "memorias":
            print(listar_memorias(args.user) or "(no recuerdo nada de este usuario)")
            continue
        if consulta.lower() == "olvidar":
            print(f"Se borraron {borrar_memorias(args.user)} memorias.")
            continue

        respuesta = await preguntar(graph, consulta, args.user, thread_id)
        print(f"\nAsistente: {respuesta}")


if __name__ == "__main__":
    asyncio.run(main())
