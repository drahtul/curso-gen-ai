"""Servidor MCP para TechStore utilizando FastMCP.

Expone herramientas de consulta de productos, precios, stock y categorías,
además de un recurso con la política de garantías y un prompt de asesor de ventas.
"""

from typing import Any
from fastmcp import FastMCP

# Inicialización del servidor MCP
mcp = FastMCP("techstore")

# Base de datos de productos en memoria
PRODUCTOS = [
    {
        "id": "PROD-001",
        "nombre": "Teclado mecánico Logitech G413",
        "categoria": "Teclados",
        "precio": 3200.0,
        "stock": 15,
        "descripcion": "Teclado mecánico retroiluminado con interruptores Romer-G de alta precisión.",
    },
    {
        "id": "PROD-002",
        "nombre": "Logitech G305",
        "categoria": "Mouses",
        "precio": 1800.0,
        "stock": 25,
        "descripcion": "Mouse inalámbrico para juegos con tecnología Lightspeed y sensor HERO.",
    },
    {
        "id": "PROD-003",
        "nombre": "Monitor LG UltraGear 27",
        "categoria": "Monitores",
        "precio": 12500.0,
        "stock": 8,
        "descripcion": "Monitor gaming de 27 pulgadas IPS QHD a 144Hz con compatibilidad G-Sync.",
    },
    {
        "id": "PROD-004",
        "nombre": "Monitor Samsung 24 FHD",
        "categoria": "Monitores",
        "precio": 4500.0,
        "stock": 0,
        "descripcion": "Monitor LED Full HD de 24 pulgadas con panel IPS y diseño sin bordes (sin stock actual).",
    },
    {
        "id": "PROD-005",
        "nombre": "Monitor Dell 27 4K",
        "categoria": "Monitores",
        "precio": 18000.0,
        "stock": 5,
        "descripcion": "Monitor profesional 4K UHD de 27 pulgadas con cobertura de color sRGB 99%.",
    },
    {
        "id": "PROD-006",
        "nombre": "Mouse Razer DeathAdder Essential",
        "categoria": "Mouses",
        "precio": 800.0,
        "stock": 30,
        "descripcion": "Mouse ergonómico con sensor óptico de 6.400 DPI.",
    },
    {
        "id": "PROD-007",
        "nombre": "Auriculares HyperX Cloud II",
        "categoria": "Audio",
        "precio": 4200.0,
        "stock": 12,
        "descripcion": "Auriculares para gaming con sonido envolvente virtual 7.1 y almohadillas de memoria.",
    },
    {
        "id": "PROD-008",
        "nombre": "Notebook Lenovo ThinkPad E14",
        "categoria": "Notebooks",
        "precio": 35000.0,
        "stock": 4,
        "descripcion": "Portátil corporativo con procesador Intel Core i7, 16GB RAM y 512GB SSD.",
    },
]


# ==========================================
# Herramientas (Tools) del Servidor MCP
# ==========================================

@mcp.tool()
async def buscar_producto(nombre: str) -> list[dict[str, Any]]:
    """Busca productos en el catálogo de TechStore por coincidencia en el nombre o categoría.

    Args:
        nombre: Término de búsqueda para filtrar productos (ej: 'Logitech', 'Monitor', 'G413').
    """
    termino = nombre.lower().strip()
    resultados = [
        p for p in PRODUCTOS
        if termino in p["nombre"].lower() or termino in p["categoria"].lower()
    ]
    return resultados


@mcp.tool()
async def consultar_precio(nombre_producto: str) -> dict[str, Any]:
    """Consulta el precio exacto de un producto específico en TechStore.

    Args:
        nombre_producto: Nombre o parte del nombre del producto a consultar.
    """
    termino = nombre_producto.lower().strip()
    coincidencias = [p for p in PRODUCTOS if termino in p["nombre"].lower()]

    if not coincidencias:
        return {"error": f"No se encontró ningún producto que coincida con '{nombre_producto}'."}

    # Si hay coincidencia exacta o tomamos la primera más relevante
    producto = coincidencias[0]
    return {
        "id": producto["id"],
        "nombre": producto["nombre"],
        "precio": producto["precio"],
        "moneda": "USD",
        "coincidencias_totales": len(coincidencias),
    }


@mcp.tool()
async def consultar_stock(nombre_producto: str) -> dict[str, Any]:
    """Consulta la disponibilidad de stock de un producto en TechStore.

    Args:
        nombre_producto: Nombre o parte del nombre del producto a consultar.
    """
    termino = nombre_producto.lower().strip()
    coincidencias = [p for p in PRODUCTOS if termino in p["nombre"].lower()]

    if not coincidencias:
        return {"error": f"No se encontró ningún producto que coincida con '{nombre_producto}'."}

    producto = coincidencias[0]
    disponible = producto["stock"] > 0
    return {
        "id": producto["id"],
        "nombre": producto["nombre"],
        "stock_disponible": producto["stock"],
        "en_stock": disponible,
        "categoria": producto["categoria"],
    }


@mcp.tool()
async def listar_categorias() -> list[str]:
    """Lista todas las categorías de productos disponibles en TechStore."""
    categorias = sorted(list(set(p["categoria"] for p in PRODUCTOS)))
    return categorias


# ==========================================
# Recurso (Resource) del Servidor MCP
# ==========================================

@mcp.resource("techstore://politica-garantias")
async def politica_garantias() -> str:
    """Condiciones generales de garantía de los productos vendidos por TechStore."""
    return """
POLÍTICA GENERAL DE GARANTÍAS - TECHSTORE
------------------------------------------
1. Cobertura:
   - Todos nuestros productos cuentan con una garantía oficial mínima de 12 meses a partir de la fecha de compra.
   - La garantía cubre exclusivamente defectos de fabricación y fallas de hardware en condiciones normales de uso.

2. Exclusiones:
   - Daños físicos ocasionados por golpes, caídas o manipulación indebida.
   - Daños por sobretensión eléctrica o exposición a líquidos/humedad.
   - Alteración, modificación o apertura no autorizada del producto.

3. Proceso de reclamo:
   - Presentar factura de compra o comprobante electrónico.
   - El producto debe entregarse con sus accesorios originales y embalaje.
   - Tiempo estimado de revisión técnica y resolución: 5 a 7 días hábiles.
"""


# ==========================================
# Prompt Reutilizable del Servidor MCP
# ==========================================

@mcp.prompt("asesor_ventas")
async def asesor_ventas_prompt(nombre_cliente: str = "") -> str:
    """Prompt reutilizable para orientar al modelo a responder como un asesor comercial profesional y cordial."""
    saludo = f" al cliente {nombre_cliente}" if nombre_cliente else ""
    return f"""Eres un Asesor de Ventas experto de TechStore.
Tu objetivo es orientar y ayudar cordialmente{saludo} con sus consultas sobre productos, precios, disponibilidad, compras y costos de envío.

Instrucciones de comportamiento:
- Mantén siempre un tono profesional, amable, servicial y comercial.
- Cuando te consulten por productos, precios o stock, utiliza las herramientas del servidor MCP para brindar información precisa.
- Cuando se requiera calcular totales de compra o costos de envío, utiliza las herramientas locales de cálculo.
- Si un producto no tiene stock, sé transparente y ofrece alternativas si están disponibles en la misma categoría.
- Brinda respuestas claras, estructuradas y con los montos debidamente formateados.
"""


def main():
    print("Iniciando servidor MCP de TechStore...")
    mcp.run()


if __name__ == "__main__":
    main()