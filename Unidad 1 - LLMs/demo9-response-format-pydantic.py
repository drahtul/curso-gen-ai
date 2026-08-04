from dotenv import load_dotenv
import os
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

class Estudiante(BaseModel):
    nombre: str
    edad: int
    carrera: str

response = client.responses.parse(
    model="gpt-4o-mini",
    input="Dame los datos de un estudiante",
    text_format=Estudiante,
)

estudiante = response.output_parsed  # instancia de Estudiante, ya validada
print(estudiante)
print(estudiante.nombre)