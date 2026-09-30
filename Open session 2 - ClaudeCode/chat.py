"""
Chatbot de consola para probar el agente a mano, igual en espíritu al
chatbot.py de Open session 1.

    python chat.py
"""

from agent import preguntar


def main():
    # La consola solo es la interfaz; el razonamiento, el routing y las tools
    # viven en agent.py y se ejecutan mediante el grafo LangGraph.
    print("Agente multidominio (películas, libros, recetas, clima, países).")
    print("Escribí 'salir' para terminar.\n")

    while True:
        mensaje = input("Vos: ").strip()

        if not mensaje:
            continue
        if mensaje.lower() in ("salir", "exit", "quit"):
            print("¡Hasta luego!")
            break

        respuesta = preguntar(mensaje)
        print(f"Bot: {respuesta}\n")


if __name__ == "__main__":
    main()
