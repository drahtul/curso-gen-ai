"""Corre los casos de prueba mínimos de la letra.

    python casos_prueba.py            # todos los casos
    python casos_prueba.py simples    # un grupo: simples | combinada | corto | largo | fuera

El caso de memoria de largo plazo lanza DOS procesos de Python distintos:
el primero aprende un dato, termina, y el segundo (thread nuevo, proceso nuevo,
checkpointer vacío) tiene que usarlo sin que el usuario lo repita.
"""
import asyncio
import subprocess
import sys

from main import crear_sistema, preguntar
from memoria import borrar_memorias

USUARIO_PRUEBA = "usuario-prueba-os3"

CASOS = {
    "simples": [
        ("Delegación simple - RAG", ["¿Qué película me recomendás sobre viajes en el espacio?"]),
        ("Delegación simple - tools propias", ["¿Cómo está el clima ahora en Montevideo?"]),
        ("Delegación simple - MCP", ["Buscá en GitHub el repositorio langchain-ai/langgraph y decime cuántas estrellas tiene."]),
    ],
    "combinada": [
        ("Delegación combinada - tool propia -> MCP",
         ["¿Cuál es la capital de Uruguay? Buscá en GitHub repositorios relacionados con esa ciudad."]),
    ],
    "corto": [
        ("Memoria de corto plazo", [
            "¿Cuál es la moneda de Brasil?",
            "¿Y cómo está el clima en Madrid?",
            "¿Cuál era el país por el que te pregunté primero, y cuál es su capital?",
        ]),
    ],
    "fuera": [
        ("Fuera de alcance", ["¿Quién ganó el mundial de fútbol de 1986?"]),
    ],
}

SESION_1 = "Hola, soy vegetariano y me encantan las películas de ciencia ficción."
SESION_2 = "¿Qué receta me recomendás para la cena?"


async def correr(grupos: list[str]) -> None:
    graph = await crear_sistema()
    for grupo in grupos:
        for titulo, turnos in CASOS[grupo]:
            thread = f"prueba-{grupo}-{titulo}"
            print("\n" + "#" * 80 + f"\n# {titulo}\n" + "#" * 80)
            for consulta in turnos:
                print(f"\nUsuario: {consulta}")
                respuesta = await preguntar(graph, consulta, USUARIO_PRUEBA, thread)
                print(f"\nAsistente: {respuesta}")


async def sesion_unica(consulta: str, thread: str) -> None:
    graph = await crear_sistema()
    print(f"\nUsuario: {consulta}")
    print(f"\nAsistente: {await preguntar(graph, consulta, USUARIO_PRUEBA, thread)}")


def caso_largo_plazo() -> None:
    print("\n" + "#" * 80 + "\n# Memoria semántica de largo plazo (2 procesos)\n" + "#" * 80)
    print(f"Se borran memorias previas de '{USUARIO_PRUEBA}': {borrar_memorias(USUARIO_PRUEBA)}")
    for n, consulta in ((1, SESION_1), (2, SESION_2)):
        print(f"\n===== PROCESO {n} (thread 'sesion-{n}') =====")
        subprocess.run([sys.executable, __file__, f"_sesion{n}"], check=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "todos"
    if arg == "_sesion1":
        asyncio.run(sesion_unica(SESION_1, "sesion-1"))
    elif arg == "_sesion2":
        asyncio.run(sesion_unica(SESION_2, "sesion-2"))
    elif arg == "largo":
        caso_largo_plazo()
    elif arg == "todos":
        asyncio.run(correr(list(CASOS)))
        caso_largo_plazo()
    else:
        asyncio.run(correr([arg]))
