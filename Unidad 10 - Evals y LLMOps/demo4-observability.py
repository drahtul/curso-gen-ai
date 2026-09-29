# Esta demo se presenta fuera de este archivo; aquí no hay flujo Python
# ejecutable. La observabilidad real se encuentra en app.py y su despliegue.
# Conviene distinguir logs (eventos), métricas (latencia, tokens y coste) y
# trazas (secuencia completa de pasos y herramientas del agente).
# Medir todos los turnos es importante porque una pregunta puede generar varias
# llamadas al LLM. timeout, max_retries y MAX_STEPS son controles operativos,
# no métricas de calidad de la respuesta.
# Ver demostración de despliegue y observabilidad