from sentence_transformers import SentenceTransformer  # type: ignore
import os
from dotenv import load_dotenv
from huggingface_hub import login
from pinecone import Pinecone  # type: ignore

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise RuntimeError("Falta la variable de entorno PINECONE_API_KEY")

INDEX_NAME = "obli-rag-cine"
BATCH_SIZE = 20

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 500 películas nuevas generadas a partir de combinaciones de datos descriptivos.
genres = [
    "science fiction",
    "mystery",
    "adventure",
    "historical drama",
    "romantic comedy",
    "thriller",
    "fantasy",
    "documentary",
    "animation",
    "crime drama",
]

settings = [
    "a floating city above the Pacific",
    "a remote village in Patagonia",
    "a crowded subway beneath New York",
    "a research station on Mars",
    "a quiet island in the Mediterranean",
    "a futuristic Nairobi",
    "the mountains of northern Japan",
    "a hidden library in Prague",
    "a desert town after a solar storm",
    "a theater in old Buenos Aires",
]

protagonists = [
    "a forensic linguist",
    "a retired astronaut",
    "a young cartographer",
    "a street photographer",
    "an ambitious botanist",
    "a piano maker",
    "a reluctant detective",
    "a marine biologist",
    "a traveling chef",
    "a museum archivist",
]

conflicts = [
    "must uncover a secret before an approaching storm arrives",
    "finds evidence that changes everything they know about their family",
    "is forced to choose between personal freedom and collective survival",
    "follows a mysterious signal that no one else can hear",
    "tries to repair an old friendship while solving an impossible puzzle",
    "discovers that a routine assignment hides a dangerous conspiracy",
    "protects a community threatened by a powerful corporation",
    "returns home to confront a forgotten promise",
    "learns that time is running out to prevent a historic disaster",
    "joins unlikely allies on a journey across unfamiliar territory",
]

movies = []
for movie_number in range(1, 501):
    genre = genres[(movie_number - 1) % len(genres)]
    setting = settings[((movie_number - 1) // len(genres)) % len(settings)]
    protagonist = protagonists[((movie_number - 1) // (len(genres) * len(settings))) % len(protagonists)]
    conflict = conflicts[(movie_number - 1) % len(conflicts)]
    year = 1980 + ((movie_number * 7) % 45)
    rating = round(6.0 + ((movie_number * 13) % 31) / 10, 1)

    movies.append(
        {
            "id": f"new-movie-{movie_number:03d}",
            "title": f"New Horizon {movie_number:03d}",
            "text": (
                f"A {genre} story about {protagonist} in {setting} who {conflict}."
            ),
            "rating": rating,
            "year": year,
        }
    )

vectors = []
for movie in movies:
    embedding = model.encode(movie["text"]).tolist()
    vectors.append(
        {
            "id": movie["id"],
            "values": embedding,
            "metadata": {
                "title": movie["title"],
                "text": movie["text"],
                "rating": float(movie["rating"]),
                "year": str(movie["year"]),
            },
        }
    )

for start in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[start:start + BATCH_SIZE]
    index.upsert(vectors=batch)
    end = min(start + BATCH_SIZE, len(vectors))
    print(f"Insertados {start + 1} a {end} registros")

print(f"{len(vectors)} películas nuevas insertadas en Pinecone")
print(index.describe_index_stats())
