from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Los tres primeros casos son ejemplos few-shot: muestran entradas y salidas
# para que el modelo induzca el patron de clasificacion.
prompt = """Clasifica el sentimiento del siguiente texto como positivo, negativo o neutro:

Texto: "La película fue increíble, me encantó"
Sentimiento: Positivo

Texto: "No me gustó el producto, llegó roto"
Sentimiento: Negativo

Texto: "El lugar está bien, nada especial"
Sentimiento: Neutro

Texto: "El servicio fue lento y la comida estaba fría"
# Este es el caso que debe resolverse aplicando el patron observado arriba.
Sentimiento:

"""

response = client.responses.create(
        # La clasificacion esta guiada por ejemplos, pero la API no restringe
        # formalmente la salida a una de las tres etiquetas.
        model="gpt-4.1-nano",
        input=prompt,
        max_output_tokens=100
)

print(response.output[0].content[0].text)
