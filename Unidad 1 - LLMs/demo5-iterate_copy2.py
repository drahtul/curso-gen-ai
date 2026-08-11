from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

instrucciones = "Continua el texto del usuario. No respondas, no expliques, no repitas lo ya escrito."

texto = "La biblioteca del pueblo guardaba un libro que"
print("Texto inicial:")
print(texto)
print("-" * 60)

for i in range(1, 9):
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instrucciones,
        input=texto,
        max_output_tokens=16,
        top_logprobs=1,
        include=["message.output_text.logprobs"]
    )

    # Extraemos el nuevo fragmento generado
    fragmento_nuevo = response.output_text.strip()

    # Añadimos al texto acumulado
    if fragmento_nuevo:
        # Añadimos un espacio si es necesario
        if not texto.endswith((" ", "\n")) and not fragmento_nuevo.startswith((" ", "\n", ".", ",", "!", "?", ";", ":")):
            texto += " "
        texto += fragmento_nuevo

    # Imprimimos la información de la iteración
    print(f"Iteración {i}")
    print(f"Fragmento nuevo: {fragmento_nuevo}")
    print(f"Texto acumulado completo: {texto}")
    print("-" * 60)