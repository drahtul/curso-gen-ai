from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

instrucciones = "Continua el texto del usuario. No respondas, no expliques, no repitas lo ya escrito."

# texto = "Habia una vez un dragon"
texto = "La biblioteca del pueblo guardaba un libro que"
print(f"Texto inicial: {texto}\n")

for i in range(10):
    # Cada iteracion es una nueva llamada al modelo. Se reenvia el texto
    # acumulado para que la proxima prediccion tenga todo el contexto previo.
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instrucciones,
        input=texto,
        max_output_tokens=16,
        top_logprobs=1,
        include=["message.output_text.logprobs"]
    )

    logprobs = response.output[0].content[0].logprobs
    # Los logprobs permiten inspeccionar los tokens candidatos, pero el corte
    # siguiente usa el texto del token como una heuristica de palabra.
    # print(logprobs)

    tokens = []
    for lp in logprobs:
        # Un espacio suele marcar el comienzo de otra palabra, aunque los
        # tokens no coinciden siempre con palabras completas.
        if tokens and lp.token.startswith(" "):
            break
        tokens.append(lp.token)

    # fragmento_nuevo agrupa tokens hasta ese corte; no equivale
    # necesariamente a una palabra segun las reglas del lenguaje natural.
    fragmento_nuevo = "".join(tokens).strip()

    if not fragmento_nuevo:
        break
    # Reenviar texto completo en cada vuelta facilita el ejemplo, pero aumenta
    # el consumo acumulado de tokens con cada nueva llamada.
    texto += " " + fragmento_nuevo
    
    print(f"Iteración {i+1}")
    print(f"Fragmento nuevo: {fragmento_nuevo} = {len(tokens)} token(s) {[t for t in tokens]}")
    print(f"Texto acumulado: {texto}\n")
