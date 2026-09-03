from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Annotated

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHART_DIR = Path(__file__).resolve().parent / "charts"
MAX_SQL_ROWS = 200
MAX_CHART_POINTS = 50


SCHEMA_SQL = """
CREATE TABLE productos (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    categoria TEXT NOT NULL,
    precio REAL NOT NULL
);

CREATE TABLE ventas (
    id INTEGER PRIMARY KEY,
    producto_id INTEGER NOT NULL REFERENCES productos(id),
    region TEXT NOT NULL,
    mes TEXT NOT NULL,
    unidades INTEGER NOT NULL,
    total REAL NOT NULL
);
"""


MOCK_PRODUCTOS = [
    (1, "Laptop Pro 14", "Computo", 1450.0),
    (2, "Monitor 27 4K", "Computo", 380.0),
    (3, "Teclado mecanico", "Accesorios", 95.0),
    (4, "Mouse ergonomico", "Accesorios", 60.0),
    (5, "Silla ergonomica", "Mobiliario", 520.0),
    (6, "Escritorio ajustable", "Mobiliario", 740.0),
]


# Ventas agrupadas por producto, mes y region.
UNIDADES_POR_PRODUCTO = {
    1: [[12, 8, 15], [18, 11, 14], [21, 9, 19]],
    2: [[30, 22, 27], [26, 25, 31], [35, 28, 33]],
    3: [[54, 41, 60], [62, 47, 58], [71, 52, 66]],
    4: [[80, 63, 77], [88, 70, 81], [95, 74, 90]],
    5: [[9, 6, 11], [13, 7, 12], [16, 10, 14]],
    6: [[5, 3, 7], [8, 4, 9], [10, 6, 12]],
}

MESES = ("2025-01", "2025-02", "2025-03")
REGIONES = ("Norte", "Sur", "Centro")

_connection: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    """Crea y reutiliza la conexión SQLite de la demo."""
    global _connection

    if _connection is not None:
        return _connection

    print("[db] Abriendo conexion a la base de datos mock...")

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_SQL)

    connection.executemany(
        "INSERT INTO productos VALUES (?, ?, ?, ?)",
        MOCK_PRODUCTOS,
    )

    precios = {product[0]: product[3] for product in MOCK_PRODUCTOS}
    ventas = []
    venta_id = 1

    for product_id, monthly_sales in UNIDADES_POR_PRODUCTO.items():
        for month, regional_sales in zip(MESES, monthly_sales):
            for region, units in zip(REGIONES, regional_sales):
                total = round(units * precios[product_id], 2)
                ventas.append(
                    (venta_id, product_id, region, month, units, total)
                )
                venta_id += 1

    connection.executemany(
        "INSERT INTO ventas VALUES (?, ?, ?, ?, ?, ?)",
        ventas,
    )
    connection.commit()

    # Impide operaciones de escritura una vez cargados los datos.
    connection.execute("PRAGMA query_only = ON")

    _connection = connection

    print(
        f"[db] Conexion lista: {len(MOCK_PRODUCTOS)} productos, "
        f"{len(ventas)} ventas."
    )

    return connection


def is_read_only_query(sql_query: str) -> bool:
    """Acepta SELECT y consultas WITH que terminen ejecutando una lectura."""
    return bool(re.match(r"^\s*(select|with)\b", sql_query, re.IGNORECASE))

# Tool 1: consultar la base de datos relacional
@tool
def query_sales_database(sql_query: str) -> str:
    """
    Ejecuta una consulta SQL de solo lectura sobre la base de ventas.

    Tablas disponibles:

    productos(id, nombre, categoria, precio)
    ventas(id, producto_id, region, mes, unidades, total)

    Solo se permiten SELECT o WITH de lectura.
    Devuelve como maximo 200 filas.
    """

    cleaned = sql_query.strip()

    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    if not cleaned:
        return "Error: la consulta esta vacia."

    if not is_read_only_query(cleaned):
        return "Error: solo se permiten consultas SELECT o WITH de lectura."

    if ";" in cleaned:
        return "Error: solo se permite una sentencia por consulta."

    try:
        cursor = get_connection().execute(cleaned)
        rows = [dict(row) for row in cursor.fetchmany(MAX_SQL_ROWS)]
    except sqlite3.Error as exc:
        return f"Error de SQL: {exc}"

    if not rows:
        return "La consulta no devolvio filas."

    return json.dumps(
        {
            "row_count": len(rows),
            "truncated": len(rows) == MAX_SQL_ROWS,
            "rows": rows,
        },
        ensure_ascii=False,
    )


SERIES_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dedcd6"
SURFACE = "#fcfcfb"

# Tool 2: generar graficas a partir de datos estructurados
@tool
def generate_chart(
    chart_type: str,
    labels: list[str],
    values: list[float],
    title: str,
    x_label: str = "",
    y_label: str = "",
    filename: str = "chart.png",
) -> str:
    """
    Genera una grafica PNG.

    chart_type debe ser 'bar', 'barh' o 'line'.
    """

    if chart_type not in {"bar", "barh", "line"}:
        return "Error: chart_type debe ser 'bar', 'barh' o 'line'."

    if not labels:
        return "Error: no se recibieron datos."

    if len(labels) != len(values):
        return "Error: labels y values deben tener la misma longitud."

    if len(labels) > MAX_CHART_POINTS:
        return f"Error: la grafica admite como maximo {MAX_CHART_POINTS} puntos."

    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in values
    ):
        return "Error: todos los valores deben ser numeros finitos."

    safe_filename = Path(filename).name
    if not safe_filename.lower().endswith(".png"):
        safe_filename += ".png"

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHART_DIR / safe_filename

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)

    try:
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        value_labels = [
            f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}"
            for value in values
        ]

        if chart_type == "bar":
            bars = ax.bar(
                labels,
                values,
                color=SERIES_COLOR,
                width=0.62,
            )
            ax.bar_label(
                bars,
                labels=value_labels,
                padding=4,
                color=TEXT_SECONDARY,
                fontsize=9,
            )
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)

        elif chart_type == "barh":
            bars = ax.barh(
                labels,
                values,
                color=SERIES_COLOR,
                height=0.62,
            )
            ax.bar_label(
                bars,
                labels=value_labels,
                padding=4,
                color=TEXT_SECONDARY,
                fontsize=9,
            )
            ax.invert_yaxis()
            ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)

        else:
            ax.plot(
                labels,
                values,
                color=SERIES_COLOR,
                linewidth=2,
                marker="o",
                markersize=6,
            )
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)

        ax.set_axisbelow(True)
        ax.set_title(
            title,
            color=TEXT_PRIMARY,
            fontsize=13,
            pad=14,
            loc="left",
        )
        ax.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=10)
        ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=10)
        ax.tick_params(
            colors=TEXT_SECONDARY,
            labelsize=9,
            length=0,
        )

        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        ax.spines["bottom"].set_color(GRID_COLOR)

        value_axis = ax.xaxis if chart_type == "barh" else ax.yaxis
        value_axis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:,.0f}")
        )

        fig.tight_layout()
        fig.savefig(output_path, facecolor=SURFACE)

    except (OSError, ValueError, RuntimeError) as exc:
        return f"Error al generar la grafica: {exc}"

    finally:
        plt.close(fig)

    print(f"[chart] Grafica guardada en {output_path}")
    return f"Grafica generada correctamente en: {output_path}"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """
Eres un analista de datos especializado en ventas.

Herramientas disponibles:
- query_sales_database: ejecuta SQL de solo lectura.
- generate_chart: genera graficas PNG.

Reglas:
1. Nunca inventes cifras.
2. Consulta la base de datos antes de responder con datos.
3. Usa GROUP BY, SUM, ORDER BY y LIMIT en SQL.
4. Para graficas, consulta primero y usa exactamente los valores obtenidos.
5. Usa line para evolucion temporal.
6. Usa bar para pocas categorias.
7. Usa barh para etiquetas largas.
8. Finaliza siempre con un resumen breve.
"""


llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-nano",
    max_tokens=1024,
    timeout=30,
    max_retries=2,
)

tools = [query_sales_database, generate_chart]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

# No se añade una salida directa de agent a END.
# tools_condition ya decide si continuar o terminar.
graph = graph_builder.compile()


def run_agent(user_input: str) -> None:
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
    }

    for step in graph.stream(
        initial_state,
        config={"recursion_limit": 8},
    ):
        node_name = next(iter(step))
        last_message = step[node_name]["messages"][-1]

        print(
            f"\n--- Nodo: {node_name} "
            f"({type(last_message).__name__}) ---"
        )

        if getattr(last_message, "tool_calls", None):
            for call in last_message.tool_calls:
                print(f"Tool: {call['name']}")
                print(f"Args: {call['args']}")
        else:
            content = last_message.content

            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )

            print(content)

    print()


def main() -> None:
    while True:
        user_query = input("Enter your query: ").strip()

        if user_query.lower() == "exit":
            print("Goodbye!")
            return

        if not user_query:
            print("Please enter a valid query.\n")
            continue

        run_agent(user_query)


if __name__ == "__main__":
    main()