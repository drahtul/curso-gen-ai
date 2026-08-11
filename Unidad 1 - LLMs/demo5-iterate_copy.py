from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

instrucciones = "Continua el texto del usuario. No respondas, no expliques, no repitas lo ya escrito."

texto = "La biblioteca del pueblo guardaba un libro que"
print(f"Texto inicial: {texto}\n")

for i in range(1, 9):  # 8 iteraciones
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instrucciones,
        input=texto,
        max_output_tokens=16,
        top_logprobs=1,
        include=["message.output_text.logprobs"]
    )

    logprobs = response.output[0].content[0].logprobs

    tokens = []
    for lp in logprobs:
        if tokens and lp.token.startswith(" "):
            break
        tokens.append(lp.token)

    fragmento_nuevo = "".join(tokens).strip()

    if not fragmento_nuevo:
        break

    texto += " " + fragmento_nuevo

    print(f"Iteración {i}")
    print(f"Fragmento nuevo: {fragmento_nuevo}")
    print(f"Texto acumulado: {texto}\n")