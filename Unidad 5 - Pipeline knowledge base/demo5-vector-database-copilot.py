from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_core.documents import Document
import re
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

# region Extraccion de texto del PDF

pdf_path = "attention-is-all-you-need.pdf"
docs = pymupdf.open(pdf_path)
print("Extraccion de texto del PDF completada")

# endregion

# region Limpieza de texto

def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)

    text = re.sub(r"\n+", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("–", "-")
            .replace("—", "-")
    )

    return text.strip()

cleaned_docs = []

for page_number, page in enumerate(docs, start=1):

    cleaned_docs.append(
        Document(
            page_content=clean_text(page.get_text()),
            metadata={
                "source": pdf_path,
                "page": page_number
            }
        )
    )

print("Limpieza de texto completada")

# endregion

# region Chunking de texto

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(cleaned_docs)

print(f"Chunks: {len(chunks)}")

print("Chunking de texto completado")

# endregion

# region Configuracion de Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "papers-index"

pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Configuracion de Pinecone completada")

# endregion

# region Cargado del indice y prueba

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = pc.Index(index_name)
index_stats = index.describe_index_stats()

if index_stats["total_vector_count"] == 0:
    chunk_embeddings = embedding_model.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    vectors = []
    for chunk, embedding in zip(chunks, chunk_embeddings):
        metadata = {
            **chunk.metadata,
            "text": chunk.page_content,
        }
        vectors.append({
            "id": str(uuid4()),
            "values": embedding,
            "metadata": metadata,
        })

    index.upsert(vectors=vectors)

query = "Whats attention?"
query_embedding = embedding_model.embed_query(query)

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True,
)

retrieved_docs = []
for match in results["matches"]:
    metadata = match.get("metadata", {})
    retrieved_docs.append(
        Document(
            page_content=metadata.get("text", ""),
            metadata={
                key: value
                for key, value in metadata.items()
                if key != "text"
            },
        )
    )

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""Using the following context, answer the question.

Question:
{query}

Context:
{context}

Answer:
"""

print(prompt)

print("Cargado del indice y prueba completada")

# endregion
