"""
Corre automáticamente los "casos de prueba mínimos" de la letra de la
Open Session 2, mostrando en cada caso qué tool eligió el agente (y con
qué argumentos) antes de dar la respuesta final. Sirve como evidencia de
que la selección de herramientas es automática.

    python test_cases.py
"""

from agent import preguntar_con_detalle

CASOS = [
    ("Base vectorial", "Recomiéndame una película de acción."),
    ("Base vectorial", "¿Quién escribió Orgullo y prejuicio?"),
    ("Base vectorial", "¿Cómo preparo brownies?"),
    ("Tool del clima", "¿Cuál es la temperatura actual en Buenos Aires?"),
    ("Tool de países", "¿Cuál es la capital de Noruega?"),
    ("Uso combinado", "¿Cómo está el clima en la capital de Francia?"),
    ("Fuera de alcance", "¿Quién ganó el Mundial de Fútbol de 1998?"),
]


def main():
    for categoria, pregunta in CASOS:
        print(f"\n{'=' * 80}\n[{categoria}] {pregunta}\n{'=' * 80}")
        preguntar_con_detalle(pregunta)


if __name__ == "__main__":
    main()
