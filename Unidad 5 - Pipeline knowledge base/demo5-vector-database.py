from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore 
from langchain_core.documents import Document
import re
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

print("Configuración de Pinecone completada")

# endregion

# region Cargado del índice y prueba

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = pc.Index(index_name)
index_stats = index.describe_index_stats()

if index_stats["total_vector_count"] > 0:
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embedding_model,
        index_name=index_name
    )
else:
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name
    )

query = "Whats attention?"

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""Using the following context, answer the question.

Question:
{query}

Context:
{context}

Answer:
"""

print(prompt)

print("Cargado del índice y prueba completada")

# endregion