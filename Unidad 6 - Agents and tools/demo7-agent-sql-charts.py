from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, List
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: guardamos la grafica a disco
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sqlite3
import json
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")


SCHEMA_SQL = """
CREATE TABLE productos (
    id        INTEGER PRIMARY KEY,
    nombre    TEXT    NOT NULL,
    categoria TEXT    NOT NULL,
    precio    REAL    NOT NULL
);

CREATE TABLE ventas (
    id          INTEGER PRIMARY KEY,
    producto_id INTEGER NOT NULL REFERENCES productos(id),
    region      TEXT    NOT NULL,
    mes         TEXT    NOT NULL,
    unidades    INTEGER NOT NULL,
    total       REAL    NOT NULL
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

MOCK_VENTAS_RAW = [
    (1, "Norte", "2025-01", 12), (1, "Sur", "2025-01", 8),  (1, "Centro", "2025-01", 15),
    (1, "Norte", "2025-02", 18), (1, "Sur", "2025-02", 11), (1, "Centro", "2025-02", 14),
    (1, "Norte", "2025-03", 21), (1, "Sur", "2025-03", 9),  (1, "Centro", "2025-03", 19),
    (2, "Norte", "2025-01", 30), (2, "Sur", "2025-01", 22), (2, "Centro", "2025-01", 27),
    (2, "Norte", "2025-02", 26), (2, "Sur", "2025-02", 25), (2, "Centro", "2025-02", 31),
    (2, "Norte", "2025-03", 35), (2, "Sur", "2025-03", 28), (2, "Centro", "2025-03", 33),
    (3, "Norte", "2025-01", 54), (3, "Sur", "2025-01", 41), (3, "Centro", "2025-01", 60),
    (3, "Norte", "2025-02", 62), (3, "Sur", "2025-02", 47), (3, "Centro", "2025-02", 58),
    (3, "Norte", "2025-03", 71), (3, "Sur", "2025-03", 52), (3, "Centro", "2025-03", 66),
    (4, "Norte", "2025-01", 80), (4, "Sur", "2025-01", 63), (4, "Centro", "2025-01", 77),
    (4, "Norte", "2025-02", 88), (4, "Sur", "2025-02", 70), (4, "Centro", "2025-02", 81),
    (4, "Norte", "2025-03", 95), (4, "Sur", "2025-03", 74), (4, "Centro", "2025-03", 90),
    (5, "Norte", "2025-01", 9),  (5, "Sur", "2025-01", 6),  (5, "Centro", "2025-01", 11),
    (5, "Norte", "2025-02", 13), (5, "Sur", "2025-02", 7),  (5, "Centro", "2025-02", 12),
    (5, "Norte", "2025-03", 16), (5, "Sur", "2025-03", 10), (5, "Centro", "2025-03", 14),
    (6, "Norte", "2025-01", 5),  (6, "Sur", "2025-01", 3),  (6, "Centro", "2025-01", 7),
    (6, "Norte", "2025-02", 8),  (6, "Sur", "2025-02", 4),  (6, "Centro", "2025-02", 9),
    (6, "Norte", "2025-03", 10), (6, "Sur", "2025-03", 6),  (6, "Centro", "2025-03", 12),
]

_connection = None


def get_connection() -> sqlite3.Connection:
    """
    Simula la conexion a la base de datos corporativa.

    En un caso real aqui iria psycopg2.connect(...) o SQLAlchemy contra Postgres,
    MySQL, etc. Para la demo usamos SQLite en memoria con datos mock y
    mantenemos una unica conexion reutilizable (patron singleton / pool de 1).
    """
    global _connection
    if _connection is not None:
        return _connection

    print("[db] Abriendo conexion a la base de datos (mock)...")
    _connection = sqlite3.connect(":memory:", check_same_thread=False)
    _connection.row_factory = sqlite3.Row
    _connection.executescript(SCHEMA_SQL)
    _connection.executemany("INSERT INTO productos VALUES (?, ?, ?, ?)", MOCK_PRODUCTOS)

    precios = {p[0]: p[3] for p in MOCK_PRODUCTOS}
    ventas = [
        (i + 1, pid, region, mes, unidades, round(unidades * precios[pid], 2))
        for i, (pid, region, mes, unidades) in enumerate(MOCK_VENTAS_RAW)
    ]
    _connection.executemany("INSERT INTO ventas VALUES (?, ?, ?, ?, ?, ?)", ventas)
    _connection.commit()
    print(f"[db] Conexion lista: {len(MOCK_PRODUCTOS)} productos, {len(ventas)} ventas.")
    return _connection


# Tool 1: consultar la base de datos relacional
@tool
def query_sales_database(sql_query: str) -> str:
    """
    Ejecuta una consulta SQL de solo lectura sobre la base de datos de ventas
    y devuelve las filas en formato JSON.

    Esquema disponible (SQLite):

      productos(id INTEGER, nombre TEXT, categoria TEXT, precio REAL)
        - categoria: 'Computo', 'Accesorios', 'Mobiliario'

      ventas(id INTEGER, producto_id INTEGER, region TEXT, mes TEXT,
             unidades INTEGER, total REAL)
        - region: 'Norte', 'Sur', 'Centro'
        - mes: texto con formato 'YYYY-MM' (datos de 2025-01 a 2025-03)
        - total: unidades * precio del producto

    Solo se aceptan sentencias SELECT. Devuelve como maximo 200 filas.
    """
    cleaned = sql_query.strip().rstrip(";").strip()

    if not cleaned.lower().startswith("select"):
        return "Error: solo se permiten consultas SELECT."
    if ";" in cleaned:
        return "Error: solo se permite una sentencia por consulta."

    try:
        cursor = get_connection().execute(cleaned)
        rows = [dict(row) for row in cursor.fetchmany(200)]
    except sqlite3.Error as exc:
        return f"Error de SQL: {exc}"

    if not rows:
        return "La consulta no devolvio filas."

    return json.dumps({"row_count": len(rows), "rows": rows}, ensure_ascii=False)


# Tool 2: generar graficas a partir de datos estructurados

SERIES_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dedcd6"
SURFACE = "#fcfcfb"

@tool
def generate_chart(
    chart_type: str,
    labels: List[str],
    values: List[float],
    title: str,
    x_label: str = "",
    y_label: str = "",
    filename: str = "chart.png",
) -> str:
    """
    Genera una grafica a partir de datos estructurados y la guarda como PNG.

    Args:
        chart_type: 'bar' (comparar categorias), 'barh' (igual, con etiquetas
            largas) o 'line' (evolucion en el tiempo).
        labels: etiquetas del eje de categorias (o del eje temporal).
        values: valores numericos, uno por etiqueta, en el mismo orden.
        title: titulo de la grafica; debe nombrar la metrica y el periodo.
        x_label: etiqueta del eje X (opcional).
        y_label: etiqueta del eje Y (opcional).
        filename: nombre del archivo PNG de salida.

    Devuelve la ruta del archivo generado.
    """
    if chart_type not in ("bar", "barh", "line"):
        return f"Error: chart_type '{chart_type}' no soportado. Usa 'bar', 'barh' o 'line'."
    if len(labels) != len(values):
        return f"Error: {len(labels)} etiquetas y {len(values)} valores; deben coincidir."
    if not labels:
        return "Error: no se recibieron datos para graficar."

    os.makedirs(CHART_DIR, exist_ok=True)
    output_path = os.path.join(CHART_DIR, os.path.basename(filename))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    etiquetas_valor = [f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}" for v in values]

    if chart_type == "bar":
        bars = ax.bar(labels, values, color=SERIES_COLOR, width=0.62)
        ax.bar_label(bars, labels=etiquetas_valor, padding=4,
                     color=TEXT_SECONDARY, fontsize=9)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    elif chart_type == "barh":
        bars = ax.barh(labels, values, color=SERIES_COLOR, height=0.62)
        ax.bar_label(bars, labels=etiquetas_valor, padding=4,
                     color=TEXT_SECONDARY, fontsize=9)
        ax.invert_yaxis()
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    else:
        ax.plot(labels, values, color=SERIES_COLOR, linewidth=2, marker="o", markersize=6)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)

    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, pad=14, loc="left")
    ax.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)

    value_axis = ax.xaxis if chart_type == "barh" else ax.yaxis
    value_axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)

    print(f"[chart] Grafica guardada en {output_path}")
    return f"Grafica '{chart_type}' generada correctamente en: {output_path}"



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """Eres un analista de datos. Tienes acceso a dos herramientas:

- query_sales_database: ejecuta SQL de solo lectura sobre la base de ventas.
- generate_chart: genera una grafica PNG a partir de datos estructurados.

Reglas de trabajo:
1. Nunca inventes cifras. Si necesitas datos, consultalos con SQL primero.
2. Deja que SQL haga el trabajo pesado (GROUP BY, SUM, ORDER BY, LIMIT).
3. Cuando el usuario pida una grafica, primero consulta y luego grafica con los
   valores exactos que devolvio la consulta.
4. Elige la forma segun el trabajo del dato: 'line' para evolucion en el tiempo,
   'bar' para comparar pocas categorias, 'barh' cuando las etiquetas son largas.
5. Cierra siempre con un resumen breve en texto de lo que encontraste.
"""

llm = ChatOpenAI(
    api_key=openai_key,
    model="gpt-4.1-nano",
    max_tokens=1024,
)

tools = [query_sales_database, generate_chart]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    """Llama al LLM con el historial actual de mensajes."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()



def run_agent(user_input: str):
    """Ejecuta el agente mostrando lo que ocurre en cada nodo del grafo."""
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
    }

    for step in graph.stream(initial_state):
        node_name = list(step.keys())[0]
        last_msg = step[node_name]["messages"][-1]

        print(f"\n--- Nodo: {node_name} ({type(last_msg).__name__}) ---")
        if getattr(last_msg, "tool_calls", None):
            for call in last_msg.tool_calls:
                print(f"Tool: {call['name']}")
                print(f"Args: {call['args']}")
        else:
            content = last_msg.content
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            print(content)
    print()


if __name__ == "__main__":
    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid query.\n")
            continue

        run_agent(user_query)
    
