from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import pymupdf
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_core.documents import Document
import re
import uuid
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# region Extracción de texto del PDF

pdf_path = "attention-is-all-you-need.pdf"
docs = pymupdf.open(pdf_path)
print("Extracción de texto del PDF completada")

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

# region Configuración de Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "papers-index"
NAMESPACE = ""

pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,          # all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Configuración de Pinecone completada")

# endregion

# region Helpers (reemplazan PineconeVectorStore)

def add_documents(
    index,
    embedding_model,
    documents: List[Document],
    namespace: str = NAMESPACE,
    batch_size: int = 100,
) -> List[str]:
    """Embed + upsert. Devuelve los IDs generados."""
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata or {} for d in documents]
    vectors = embedding_model.embed_documents(texts)

    ids = [str(uuid.uuid4()) for _ in documents]
    records = [
        {
            "id": ids[i],
            "values": vectors[i],
            "metadata": {
                **metadatas[i],
                "text": texts[i],
            },
        }
        for i in range(len(documents))
    ]

    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i : i + batch_size], namespace=namespace)

    return ids


def similarity_search(
    index,
    embedding_model,
    query: str,
    k: int = 3,
    filter: Optional[dict] = None,
    namespace: str = NAMESPACE,
) -> List[Document]:
    """Equivalente a vectorstore.similarity_search / retriever."""
    query_vector = embedding_model.embed_query(query)

    results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True,
        namespace=namespace,
        filter=filter,
    )

    docs = []
    for match in results.matches:
        meta = match.metadata or {}
        text = meta.pop("text", "")
        docs.append(Document(page_content=text, metadata=meta))
    return docs

# endregion

# region Cargado del índice y retrieval

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = pc.Index(index_name)
index_stats = index.describe_index_stats()

if index_stats.get("total_vector_count", 0) == 0:
    print("Índice vacío → subiendo chunks...")
    add_documents(index, embedding_model, chunks)
    print(f"Se subieron {len(chunks)} chunks")
else:
    print(f"Índice ya tiene {index_stats['total_vector_count']} vectores")

query = "Whats attention?"

retrieved_docs = similarity_search(
    index=index,
    embedding_model=embedding_model,
    query=query,
    k=3,
)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""Using the following context, answer the question.

Question:
{query}

Context:
{context}

Answer:
"""

print("Prompt construido:")
print(prompt)
print("-" * 60)

# endregion

# region Generación de respuesta con OpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",          # o "gpt-4o", "gpt-4.1-mini", etc.
    temperature=0,
    # api_key se toma automáticamente de OPENAI_API_KEY en el .env
)

response = llm.invoke(prompt)

print("Respuesta del modelo:")
print(response.content)

print("Cargado del índice y prueba completada")

# endregion