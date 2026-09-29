from dotenv import load_dotenv
import os
import math
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

instrucciones = "Responde unicamente con UNA palabra, en minusculas y sin punto final."

preguntas = [
    # "Cual es la capital de Uruguay?",
    # "Dime una palabra esdrújula.",
    "crea un nombre para una marca de pantufla de invierno.",
    # "Inventa una palabra que no exista.",
]

for pregunta in preguntas:
    response = client.responses.create(
        model="gpt-4o-mini", # Modelo que vamos a usar
        instructions=instrucciones, # Instruccion que orienta la generacion
        input=pregunta, # Contexto que condiciona el siguiente token
        max_output_tokens=16, # Limite de tokens de salida, no de palabras
        # Devuelve el token elegido y dos alternativas por posicion.
        top_logprobs=2,
        # Solicita los logprobs para estudiar la distribucion token a token.
        include=["message.output_text.logprobs"],
        # temperature modifica la concentracion de la distribucion; top_p
        # recorta por probabilidad acumulada. Aqui top_p queda en 1.
        temperature=1,
        top_p=1
    )

    logprobs = response.output[0].content[0].logprobs

    print(f"\n--- {pregunta} ---")

    print(f"Respondio: {response.output_text!r}  ({len(logprobs)} tokens)")

    for i, logprob in enumerate(logprobs):
        certeza = math.exp(logprob.logprob)
        # Esta certeza indica la probabilidad asignada a ese token en contexto,
        # no la veracidad factual de toda la respuesta generada.
        print(f"\n  [{i}] elegido: {logprob.token!r} ({certeza:.2%})")

        for alternativa in logprob.top_logprobs:
            # el modelo trabaja en log entonces lo pasamos a probabilidad con exp()
            probabilidad = math.exp(alternativa.logprob)
            marca = "<--" if alternativa.token == logprob.token else ""
            barra = "#" * int(probabilidad * 30)
            print(f"      {alternativa.token!r:<14} {probabilidad:.2%} {barra} {marca}")
