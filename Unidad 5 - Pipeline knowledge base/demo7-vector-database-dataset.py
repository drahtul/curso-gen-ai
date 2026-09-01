import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# region Extraccion de datos del CSV

csv_path = "ai_engineering_ecosystem_intelligence.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} repositories.")

# Las filas sin descripcion no aportan informacion semantica para el embedding.
df = df.dropna(subset=["description"])
df = df[df["description"].astype(str).str.strip() != ""]

# endregion

# region Creacion de documentos

documents = []

for _, repository in df.iterrows():
    documents.append(
        Document(
            page_content=str(repository["description"]).strip(),
            metadata={
                "repo_name": str(repository["repo_name"]),
                "html_url": str(repository["html_url"]),
                "language": str(repository["language"]),
            },
        )
    )

print(f"Created {len(documents)} documents.")

# endregion

# region Configuracion de Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "ai-engineering-ecosystem-index-v1"

pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

print("Pinecone configuration completed.")

# endregion

# region Carga en la base vectorial

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = pc.Index(index_name)
index_stats = index.describe_index_stats()

if index_stats["total_vector_count"] > 0:
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embedding_model,
        index_name=index_name,
    )
    print("Existing vector index loaded.")
else:
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        index_name=index_name,
    )
    print(f"Loaded {len(documents)} documents into Pinecone.")

# endregion
