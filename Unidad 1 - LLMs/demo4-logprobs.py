from dotenv import load_dotenv
import os
import math
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

instrucciones = "Responde unicamente con UNA palabra, en minusculas y sin punto final."

preguntas = [
    "Cual es la capital de Uruguay?",
    "Dime una palabra esdrújula.",
    "Inventa una palabra que no exista.",
]

for pregunta in preguntas:
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instrucciones,
        input=pregunta,
        max_output_tokens=16,
        top_logprobs=3,
        include=["message.output_text.logprobs"]
    )

    logprobs = response.output[0].content[0].logprobs

    print(f"\n--- {pregunta} ---")

    print(f"Respondio: {response.output_text!r}  ({len(logprobs)} tokens)")

    for i, logprob in enumerate(logprobs):
        certeza = math.exp(logprob.logprob)
        print(f"\n  [{i}] elegido: {logprob.token!r} ({certeza:.2%})")

        for alternativa in logprob.top_logprobs:
            # el modelo trabaja en log entonces lo pasamos a probabilidad con exp()
            probabilidad = math.exp(alternativa.logprob)
            marca = "<--" if alternativa.token == logprob.token else ""
            barra = "#" * int(probabilidad * 30)
            print(f"      {alternativa.token!r:<14} {probabilidad:.2%} {barra} {marca}")
