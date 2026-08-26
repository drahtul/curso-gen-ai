import importlib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
from huggingface_hub import login

try:
    SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se encontró 'sentence-transformers'. Instalalo con: pip install sentence-transformers"
    ) from exc

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

# 1. Cargar el modelo
model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# model = SentenceTransformer(
#     "paraphrase-multilingual-MiniLM-L12-v2",
#     revision="main"
# )
# 2. Textos de ejemplo
texts = [
    "El gato está durmiendo en el sofá",
    "Un perro juega en el parque",
    "La inteligencia artificial está transformando el mundo",
    "Un felino descansa sobre un mueble"
]

# 3. Generar embeddings
embeddings = model.encode(texts)

# 4. Mostrar embeddings (opcional)
print("Dimensión de embeddings:", embeddings[0].shape)
print("\nEjemplo de embedding (primeros 5 valores):")
print(embeddings[3][:5])

# 5. Calcular similitud entre todos
similarity_matrix = cosine_similarity(embeddings)

print("\n📊 Matriz de similitud:")
print(np.round(similarity_matrix, 2))

# 6. Buscar similitud entre frases específicas
print("\n🔍 Similitud entre texto 1 y 4:")
print(similarity_matrix[0][3])

print("\n🔍 Similitud entre texto 1 y 2:")
print(similarity_matrix[0][1])

print("\n🔍 Similitud entre texto 2 y 3:")
print(similarity_matrix[1][2])

print(cosine_similarity([embeddings[1]], [embeddings[2]]))  # Similaridad entre el segundo y tercer texto