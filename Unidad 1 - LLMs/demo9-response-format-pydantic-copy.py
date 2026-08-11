from dotenv import load_dotenv
import os
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

class Libro(BaseModel):
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

# Muestra el nombre de la clase del objeto almacenado en 'libro'
print(type(libro).__name__)
# Convierte el objeto Pydantic en un diccionario Python para mostrarlo o usarlo fácilmente
print(libro.model_dump())