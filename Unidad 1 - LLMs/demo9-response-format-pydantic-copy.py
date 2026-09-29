from dotenv import load_dotenv
import os
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

class Libro(BaseModel):
    # Pydantic representa la salida esperada como una estructura tipada y
    # facilita su validacion antes de integrarla con el resto de la aplicacion.
    título: str
    autor: str
    género: str
    descripción_corta: str

response = client.responses.parse(
    model="gpt-4o-mini",
    input="Dame los datos de un libro",
    text_format=Libro,
)

libro: Libro = response.output_parsed

# La respuesta parseada es una instancia de Libro, no un diccionario comun.
print(type(libro).__name__)
# model_dump() serializa localmente esa instancia a un diccionario Python;
# esto ocurre despues de recibir y validar la respuesta del modelo.
print(libro.model_dump())