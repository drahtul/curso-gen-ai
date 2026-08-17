# Objetivo
  Chatbot de consola que logre:
- Responder preguntas generales de cualquier tema, como un chatbot conversacional 
normal. 
- Cuando la pregunta del usuario se refiera a un país específico, identificar de cuál se 
trata y responder usando datos reales y actualizados obtenidos de la API de países (no del 
conocimiento interno del modelo). 
- Si el país mencionado no está disponible en la fuente de datos, aclararlo en vez de 
inventar información.

# Requisitos
Arquitectura del chat:
- FASE 1 - Detección: Identificar si el mensaje del usuario se refiere a un país y a cuál. 
- FASE 2 - Inyección: Si se detectó un país, consultar la API de países y agregar esos datos como contexto adicional 
antes de generar la respuesta. 
- FASE 3 - Respuesta: Generar la respuesta final al usuario, usando el contexto inyectado como única 
fuente de verdad sobre el país. 

Técnicos:
- Lenguaje Python - OpenAi
- Generar código en demo.py
- APi de países:
  + http://localhost:3000
  + GET /countries - Lista de países
  + GET /countries/:name - Información de un país

Funcionales:
- Mantener una conversación de varios turnos, no solo responder una pregunta 
aislada.
- Dejar algún registro visible (log en consola, por ejemplo) de cuándo el chatbot detectó un país y cuándo consultó la API con el fin de poder auditar el comportamiento durante la demo

# Restricciones
- No usar function calling / tool calling de la API del modelo, ni frameworks de agentes 
(LangChain agents, etc.).