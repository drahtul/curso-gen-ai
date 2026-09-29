from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Prompt amplio: deja muchos aspectos sin especificar, por lo que el modelo
# debe inferir audiencia, profundidad, estructura y ejemplos.
prompt1 = "Hazme una clase sobre bases de datos"
# Este prompt agrega rol, audiencia, alcance, ejemplos, actividad y formato.
# Esas restricciones reducen la ambiguedad y hacen mas evaluable la respuesta.
prompt2 = """Actúa como profesor universitario de bases de datos. Necesito una clase introductoria sobre bases de datos relacionales para estudiantes de primer año.
Incluye:

una explicación clara de los conceptos principales (tabla, fila, clave primaria)
2 ejemplos simples
una actividad práctica corta para realizar en clase
Formato: estructura en secciones tipo diapositiva (título + bullets)."""

def obtener_respuesta(prompt):
    # La funcion permite comparar prompts distintos usando la misma llamada y
    # el mismo modelo; asi el foco del experimento queda en el prompt.
    response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt,
        # Es un limite maximo de longitud, no una garantia de contenido exacto.
        max_output_tokens=1000
    )
    return response.output[0].content[0].text

respuesta1 = obtener_respuesta(prompt1)
respuesta2 = obtener_respuesta(prompt2)

print("=== PROMPT 1 ===")
print("Prompt:", prompt1)
print("Respuesta:", respuesta1)

print("\n=== PROMPT 2 ===")
print("Prompt:", prompt2)
print("Respuesta:", respuesta2)