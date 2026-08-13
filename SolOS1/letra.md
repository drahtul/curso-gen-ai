# Objetivo
  Chatbot de consola que logre:
- Responder preguntas generales de cualquier tema, como un chatbot conversacional 
normal. 
- Cuando la pregunta del usuario se refiera a un país específico, identificar de cuál se 
trata y responder usando datos reales y actualizados obtenidos de la API (no del 
conocimiento interno del modelo). 
- Si el país mencionado no está disponible en la fuente de datos, aclararlo en vez de 
inventar información.

# Requisitos
Arquitectura del chat:
- FASE 1 - Detección: Identificar si el mensaje del usuario se refiere a un país y a cuál. 
- FASE 2 - Inyección: Si se detectó un país, consultar la API y agregar esos datos como contexto adicional 
antes de generar la respuesta. 
- FASE 3 - Respuesta: Generar la respuesta final al usuario, usando el contexto inyectado como única 
fuente de verdad sobre el país. 

Técnicos:
- Lenguaje Python - OpenAi
- Api de países: 

Funcionales:


# Restricciones
No usar function calling / tool calling de la API del modelo, ni frameworks de agentes 
(LangChain agents, etc.).