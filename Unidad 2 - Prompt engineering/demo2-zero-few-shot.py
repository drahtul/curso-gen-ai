from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Los tres primeros casos son ejemplos few-shot: muestran entradas y salidas
# para que el modelo induzca el patron de clasificacion.
# La clasificacion esta guiada por ejemplos, pero la API no restringe
# formalmente la salida a una de las tres etiquetas.
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

# Los dos productos completos funcionan como demostraciones few-shot de la
# estructura y del estilo que el modelo debe imitar.
# "Exactamente este formato" es una instruccion textual, no una
# validacion estructurada: el programa no comprueba sus campos.
prompt2 = """Describe el producto siguiendo exactamente este formato:

Producto: Cafetera automática
Descripción: Prepara café de forma rápida y sencilla con solo presionar un botón.
Ventaja: Ideal para ahorrar tiempo en las mañanas.

Producto: Mochila impermeable
Descripción: Protege tus pertenencias incluso en condiciones de lluvia intensa.
Ventaja: Perfecta para viajes y uso diario.

Producto: Auriculares inalámbricos con cancelación de ruido
# El modelo debe generar los campos faltantes a partir del producto y del
# patron aprendido; no se le proporciona una respuesta de ejemplo para este.
Descripción:
Ventaja:
"""
response = client.responses.create(
        model="gpt-4.1-nano",
        # input=prompt,
        input=prompt2,
        max_output_tokens=100
)

print(response.output[0].content[0].text)