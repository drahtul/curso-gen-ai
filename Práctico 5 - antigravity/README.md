# Práctico: Integración de LangGraph con MCP (TechStore)

Solución desarrollada para el ejercicio práctico de la **Unidad 8 - Model Context Protocol (MCP)** en la carpeta `practico_gemini`.

---

## 📌 Descripción del Proyecto

El proyecto implementa un asistente inteligente comercial para la empresa **TechStore**, combinando:
1. **Servidor MCP (`techstore_server.py`)**: Construido con `FastMCP`, expone catálogo de productos, consultas de precios, stock, categorías, un recurso de políticas de garantías y un prompt de asesor de ventas.
2. **Agente Inteligente con LangGraph (`techstore_agent.py`)**: Descubre dinámicamente las herramientas del servidor MCP vía transporte `stdio` y las combina con herramientas locales de cálculo de compras y cálculo de envíos.

---

## 📁 Estructura de Archivos

```
Unidad 8 - MCP/practico_gemini/
│
├── techstore_server.py    # Servidor FastMCP con Tools, Resource y Prompt
├── techstore_agent.py     # Agente LangGraph con Tools locales + MCP dinámicas
└── README.md              # Documentación y guía de ejecución
```

---

## 🛠️ Detalle de Componentes

### 1. Servidor MCP (`techstore_server.py`)

* **Tools expuestas**:
  * `buscar_producto(nombre: str)`: Búsqueda flexible por nombre o categoría dentro del catálogo.
  * `consultar_precio(nombre_producto: str)`: Retorna el precio unitario del producto consultado.
  * `consultar_stock(nombre_producto: str)`: Retorna la cantidad en stock y disponibilidad.
  * `listar_categorias()`: Retorna todas las categorías disponibles (`Audio`, `Mouses`, `Monitores`, `Notebooks`, `Teclados`).

* **Resource expuesto**:
  * `techstore://politica-garantias`: Retorna las condiciones generales de garantía de TechStore (cobertura 12 meses, exclusiones y proceso de reclamo).

* **Prompt reutilizable**:
  * `asesor_ventas(nombre_cliente: str = "")`: Plantilla estructurada para orientar al LLM a comportarse como un asesor comercial profesional, cordial y enfocado en ventas.

---

### 2. Agente LangGraph (`techstore_agent.py`)

* **Tools Locales (propias del agente)**:
  * `calcular_total(precios: list[float])`: Suma una lista de valores numéricos de productos.
  * `calcular_costo_envio(monto_compra: float)`:
    * Compras < $1,000 → Costo de envío: **$120**.
    * Compras entre $1,000 y $5,000 → Costo de envío: **$60**.
    * Compras > $5,000 → Costo de envío: **$0 (Envío gratuito)**.

* **Descubrimiento Dinámico de MCP**:
  * El agente inicia un proceso `stdio` apuntando a `techstore_server.py` utilizando `ClientSession` y `stdio_client`.
  * Invoca `load_mcp_tools(session)` para cargar dinámicamente las herramientas remotas sin acoplamiento estático.

* **Grafo de Decisión**:
  * Utiliza `StateGraph(AgentState)` con `ToolNode` y `tools_condition`.
  * El LLM decide autónomamente en cada turno si invocar una herramienta MCP, una herramienta local, ambas secuencialmente o responder directamente al usuario.

---

## 🚀 Cómo Ejecutar

Asegúrate de tener configurado tu archivo `.env` en la raíz del repositorio con tu clave de OpenAI:
```env
OPENAI_API_KEY=tu_api_key_aqui
```

Ejecuta el agente principal:
```bash
python "Unidad 8 - MCP/practico_gemini/techstore_agent.py"
```

El script ejecutará automáticamente los **4 ejemplos solicitados en la consigna**:
1. **Ejemplo 1 (Solo MCP)**: `¿Cuánto cuesta el teclado mecánico Logitech G413?`
2. **Ejemplo 2 (Solo Local)**: `Si compro un monitor de $4,500 y un mouse de $800, ¿cuánto pagaré incluyendo el envío?`
3. **Ejemplo 3 (Híbrido MCP + Local)**: `¿Cuál es el precio del Logitech G305 y cuánto pagaré si agrego el envío?`
4. **Ejemplo 4 (Combinación MCP)**: `¿Qué productos tienen stock disponible dentro de la categoría "Monitores"?`

Al finalizar los ejemplos, se ofrece un **modo interactivo de consola** para realizar consultas libres.
