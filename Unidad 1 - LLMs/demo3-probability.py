from dotenv import load_dotenv
import os
import math
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

instrucciones = "Continua el texto del usuario. No respondas, no expliques, no repitas lo ya escrito."

prompt = "El perro esta en el"
# prompt = "El perro esta en el ja"

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=instrucciones,
    input=prompt,
    max_output_tokens=16,
    # Son los 10 tokens candidatos mas probables, no necesariamente 10
    # palabras completas: la tokenizacion puede dividir una palabra.
    top_logprobs=10,
    include=["message.output_text.logprobs"]
)

# Una logprobabilidad suele estar expresada como logaritmo natural de la
# probabilidad. Aplicar exp() recupera una probabilidad entre 0 y 1.
primer_token = response.output[0].content[0].logprobs[0]

print(f"{prompt} ...")
for alternativa in primer_token.top_logprobs:
    probabilidad = math.exp(alternativa.logprob)
    # Solo mostramos candidatos principales; por eso sus probabilidades
    # visibles pueden no sumar exactamente 1: falta el resto de la cola.
    print(f"  {alternativa.token!r:<12} {probabilidad:7.2%}")
