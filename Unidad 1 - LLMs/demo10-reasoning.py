from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

prompt = "Un bate y una pelota cuestan $1.10 en total. " \
         "El bate cuesta $1.00 más que la pelota. " \
         "¿Cuánto cuesta la pelota?"

# El reasoning effort controla el presupuesto interno que el modelo puede
# dedicar al razonamiento. Puede afectar latencia, coste y precision; los
# tokens internos no son necesariamente texto visible para el usuario.
# Los valores aceptados dependen del modelo y de la version de la API.

for effort in ["none", "low", "xhigh"]:
    # Para comparar el esfuerzo de forma didactica se mantiene fijo el
    # problema y se varia solo este parametro.
    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        reasoning={"effort": effort}
    )

    print(f"\n--- Reasoning effort: {effort} ---")
    print("Respuesta:", response.output_text)
    print("Tokens de input:", response.usage.input_tokens)
    print("Tokens de output:", response.usage.output_tokens)
    print("Tokens de razonamiento:", response.usage.output_tokens_details.reasoning_tokens)
    # La metrica separa tokens visibles de razonamiento y permite estudiar el
    # coste total de resolver el mismo problema con distintos presupuestos.
    print("Tokens totales:", response.usage.total_tokens)