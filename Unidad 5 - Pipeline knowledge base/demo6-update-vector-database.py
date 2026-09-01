from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore #type: ignore
from pinecone import Pinecone
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "papers-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# endregion

# Upsert a traves de Pinecone API
def direct_upsert():
    print("Upsert a traves de Pinecone API")

    # Example: Update a single document
    new_content = "Updated content about Attention mechanisms in neural networks"
    new_embedding = embedding_model.embed_query(new_content)

    id = "06dd9f70-0859-4772-b489-c239b72c6faf"
    # Upsert: Updates if exists, inserts if doesn't exist
    index.upsert(
        vectors=[
            {
                "id": id,  # Debe hacer match con un ID ya existente
                "values": new_embedding,
                "metadata": {
                    "source": "attention-is-all-you-need.pdf",
                    "page": 10,
                    "text": new_content,
                    "updated_at": "2026-08-27"
                }
            }
        ]
    )

    print(f"Vector con ID: {id} actualizado con nueva información")



# Update por IDs a traves de ecosistema LangChain
def langchain_update_by_id():
    print("Update por IDs a traves de ecosistema LangChain")
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embedding_model,
        index_name=index_name,
    )

    updated_docs = [
        Document(page_content="Attention mechanism: ...",
                 metadata={"page": 1, "section": "attention"}),
        Document(page_content="Self-attention: ...",
                 metadata={"page": 2, "section": "self-attention"}),
    ]

    ids = ["doc-attention-002",
           "doc-attention-003"]

    result = vectorstore.add_documents(updated_docs, ids=ids)

    print(f"{len(result)} vectores actualizados: {result}")



# Update incremental con version
def versioned_updates():
    print("Update incremental con version")

    id = "06dd9f70-0859-4772-b489-c239b72c6faf"

    v1_content = "Attention mechanism: ..."

    v2_content = "Improved attention mechanism content with better explanation"
    v2_embedding = embedding_model.embed_query(v2_content)

    index.upsert(
        vectors=[
            {
                "id": id,
                "values": v2_embedding,
                "metadata": {
                    "content": v2_content,
                    "version": "2",
                    "updated_at": "2026-08-27",
                    "update_reason": "Content improvement for clarity",
                    "archived_content_v1": v1_content,
                }
            }
        ]
    )

    print(f"Vector con ID: {id} actualizado con nueva información")
