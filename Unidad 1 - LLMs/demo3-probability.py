from dotenv import load_dotenv
import os
import math
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

instrucciones = "Continua el texto del usuario. No respondas, no expliques, no repitas lo ya escrito."

prompt = "El perro esta en el"
# prompt = "El perro esta en el ja"

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=instrucciones,
    input=prompt,
    max_output_tokens=16,
    top_logprobs=10, #10 palabras más probables
    include=["message.output_text.logprobs"]
)

primer_token = response.output[0].content[0].logprobs[0]

print(f"{prompt} ...")
for alternativa in primer_token.top_logprobs:
    probabilidad = math.exp(alternativa.logprob)
    print(f"  {alternativa.token!r:<12} {probabilidad:7.2%}")
