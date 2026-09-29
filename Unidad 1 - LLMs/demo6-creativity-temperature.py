from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt = "Describe en dos frases una novela de ciencia ficción sobre un faro"

# Repetir el mismo prompt permite observar la variabilidad de muestreo para
# cada temperatura, en lugar de confundirla con cambios en la instruccion.
for temperature in [0.1, 0.5, 0.9]:
    # Una temperatura baja concentra la distribucion en tokens probables;
    # una alta permite explorar mas alternativas, pero no garantiza calidad.
    for i in range(3):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=50,
            temperature=temperature
        )
    
        print(f"\n--- temperature {temperature} - intento {i + 1} ---")
        print(response.output_text)