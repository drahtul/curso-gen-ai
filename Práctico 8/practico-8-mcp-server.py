from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("techstore")

PRODUCTOS = [
    {"id": 1, "nombre": "Teclado mecánico Logitech G413", "categoria": "Teclados", "precio": 1850.0, "stock": 12},
    {"id": 2, "nombre": "Teclado Redragon Kumara K552", "categoria": "Teclados", "precio": 950.0, "stock": 0},
    {"id": 3, "nombre": "Mouse Logitech G305", "categoria": "Mouses", "precio": 780.0, "stock": 25},
    {"id": 4, "nombre": "Mouse Razer DeathAdder Essential", "categoria": "Mouses", "precio": 650.0, "stock": 8},
    {"id": 5, "nombre": "Monitor Samsung Odyssey 27\"", "categoria": "Monitores", "precio": 4500.0, "stock": 5},
    {"id": 6, "nombre": "Monitor LG UltraGear 24\"", "categoria": "Monitores", "precio": 3200.0, "stock": 0},
    {"id": 7, "nombre": "Monitor Dell S2721DGF 27\"", "categoria": "Monitores", "precio": 6100.0, "stock": 3},
    {"id": 8, "nombre": "Auriculares HyperX Cloud II", "categoria": "Audio", "precio": 1650.0, "stock": 15},
    {"id": 9, "nombre": "Parlantes Logitech Z313", "categoria": "Audio", "precio": 890.0, "stock": 0},
    {"id": 10, "nombre": "Notebook Lenovo IdeaPad 3", "categoria": "Notebooks", "precio": 14500.0, "stock": 4},
]

POLITICA_GARANTIAS = """Política de garantías - TechStore

1. Plazo: todos los productos nuevos cuentan con 12 meses de garantía desde la fecha de compra.
   Las notebooks cuentan con 24 meses de garantía.
2. Cobertura: la garantía cubre defectos de fabricación y fallas de funcionamiento en condiciones
   normales de uso.
3. Exclusiones: no cubre daños por golpes, caídas, humedad, sobretensión, mal uso, desgaste
   natural ni equipos intervenidos por terceros no autorizados.
4. Requisitos: presentar la factura de compra y el producto con sus accesorios y, de ser posible,
   su embalaje original.
5. Derecho de arrepentimiento: el cliente dispone de 10 días corridos desde la entrega para
   devolver el producto sin uso y en su embalaje original.
6. Resolución: TechStore evaluará el producto en un plazo máximo de 15 días hábiles y procederá
   a su reparación, reemplazo o reintegro del importe según corresponda.
"""


def _buscar(nombre: str) -> list[dict]:
    termino = nombre.lower().strip()
    return [p for p in PRODUCTOS if termino in p["nombre"].lower()]


def _resolver_producto(nombre: str) -> dict:
    """Devuelve el producto si hay una única coincidencia, o un error descriptivo."""
    coincidencias = _buscar(nombre)
    if not coincidencias:
        return {"ok": False, "error": f"No se encontró ningún producto que coincida con '{nombre}'"}
    if len(coincidencias) > 1:
        return {
            "ok": False,
            "error": f"Hay varios productos que coinciden con '{nombre}', especifique mejor",
            "coincidencias": [p["nombre"] for p in coincidencias],
        }
    return {"ok": True, "producto": coincidencias[0]}


@mcp.tool()
def buscar_producto(nombre: str) -> Any:
    """Busca productos cuyo nombre contenga el texto indicado (no distingue mayúsculas).

    Args:
        nombre: Nombre o parte del nombre del producto (ej: 'G413', 'monitor', 'Logitech')
    """
    coincidencias = _buscar(nombre)
    return {"ok": True, "cantidad": len(coincidencias), "productos": coincidencias}


@mcp.tool()
def consultar_precio(nombre: str) -> Any:
    """Consulta el precio de un producto.

    Args:
        nombre: Nombre o parte del nombre del producto
    """
    resultado = _resolver_producto(nombre)
    if not resultado["ok"]:
        return resultado
    producto = resultado["producto"]
    return {"ok": True, "producto": producto["nombre"], "precio": producto["precio"]}


@mcp.tool()
def consultar_stock(nombre: str) -> Any:
    """Consulta el stock disponible de un producto.

    Args:
        nombre: Nombre o parte del nombre del producto
    """
    resultado = _resolver_producto(nombre)
    if not resultado["ok"]:
        return resultado
    producto = resultado["producto"]
    return {
        "ok": True,
        "producto": producto["nombre"],
        "stock": producto["stock"],
        "disponible": producto["stock"] > 0,
    }


@mcp.tool()
def listar_categorias() -> Any:
    """Lista todas las categorías de productos disponibles en la tienda."""
    categorias = sorted({p["categoria"] for p in PRODUCTOS})
    return {"ok": True, "categorias": categorias}


@mcp.tool()
def listar_productos_por_categoria(categoria: str) -> Any:
    """Lista los productos (con precio y stock) de una categoría.

    Args:
        categoria: Nombre de la categoría (usar listar_categorias para ver las válidas)
    """
    productos = [p for p in PRODUCTOS if p["categoria"].lower() == categoria.lower().strip()]
    if not productos:
        return {"ok": False, "error": f"La categoría '{categoria}' no existe o no tiene productos"}
    return {"ok": True, "categoria": productos[0]["categoria"], "productos": productos}


@mcp.resource("techstore://politica-garantias")
def politica_garantias() -> str:
    """Política de garantías: condiciones generales de garantía de los productos vendidos."""
    return POLITICA_GARANTIAS


@mcp.resource("techstore://categorias")
def categorias() -> str:
    """Categorías de productos disponibles (versión resource de listar_categorias)."""
    return "\n".join(f"- {c}" for c in sorted({p["categoria"] for p in PRODUCTOS}))


@mcp.prompt(name="asesor_de_ventas")
def asesor_de_ventas() -> str:
    """Asesor de ventas: orienta al modelo para responder consultas comerciales con tono profesional y cordial."""
    return (
        "Eres el asesor de ventas de TechStore, una tienda de tecnología. "
        "Responde las consultas de los clientes con un tono profesional, cordial y cercano, "
        "en español y de forma clara y concisa.\n"
        "Pautas:\n"
        "- Nunca inventes precios, stock ni productos: consulta siempre las herramientas disponibles.\n"
        "- Para cálculos (totales, envío) usa las herramientas de cálculo, no hagas cuentas de memoria.\n"
        "- Si un producto no tiene stock, indícalo con amabilidad y sugiere alternativas de la misma categoría.\n"
        "- Si la búsqueda es ambigua, muestra las opciones y pide al cliente que precise.\n"
        "- Expresa los montos con el signo $ y separador de miles.\n"
        "- Cuando corresponda, menciona la garantía según la política de la tienda."
    )


if __name__ == "__main__":
    print("MCP server TechStore corriendo en http://127.0.0.1:8000/mcp. Press Ctrl+C to stop.")
    mcp.run(transport="streamable-http")
