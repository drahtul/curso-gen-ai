# Esta demo se ejecuta desde la carpeta promptfoo, no como un programa Python.
# Promptfoo compara variantes de prompts contra los mismos casos para medir
# cambios de comportamiento de forma reproducible.
# Sus aserciones deterministas (contains, not-contains y longitud) contrastan
# señales exactas; llm-rubric usa otro LLM como juez y puede ser variable.
# Los tool_results del YAML son fixtures simulados, no llamadas reales a tools.
# Los casos tambien prueban resistencia a prompt injection y a precios
# inventados, dos riesgos habituales en aplicaciones generativas.
# Para ejecutar la demo: npx promptfoo eval desde la carpeta promptfoo.