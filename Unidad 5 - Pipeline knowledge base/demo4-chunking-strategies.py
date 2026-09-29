import pymupdf
import re

pdf_path = "attention-is-all-you-need.pdf"

def clean_text(text: str) -> str:
    # Corrige palabras partidas por el salto de linea que introdujo el PDF.
    text = re.sub(r"-\n", "", text)

    # Convierte el formato visual en texto continuo para poder segmentarlo.
    text = re.sub(r"\n+", " ", text)

    # Reduce ruido de espacios antes de aplicar las estrategias de chunking.
    text = re.sub(r"\s+", " ", text)

    # Uniforma signos equivalentes para que el texto sea mas consistente.
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("–", "-")
            .replace("—", "-")
    )

    return text.strip()

doc = pymupdf.open(pdf_path)

print(f"Document has {doc.page_count} pages.\n")

pdf_complete_text = ""

# Al unir paginas se facilita comparar estrategias sobre el mismo contenido,
# aunque se pierde la frontera explicita entre paginas del documento original.
for page in doc:
    pdf_complete_text += clean_text(page.get_text())
    pdf_complete_text += "\n\n"

# region Fixed-size chunking

def fixed_size_chunking(text, chunk_size=600):
    # Es simple y predecible, pero puede cortar frases o conceptos justo en
    # los limites y afectar la calidad del contexto recuperado.
    return [
        text[i:i+chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

fixed_chunks = fixed_size_chunking(pdf_complete_text, 600)

# endregion

# region Sliding window chunking

def sliding_window_chunking(text, chunk_size=500, overlap=100):
    # El solapamiento conserva contexto entre chunks consecutivos, a cambio de
    # repetir texto, embeddings y almacenamiento.

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

sliding_window_chunks = sliding_window_chunking(pdf_complete_text, chunk_size=500, overlap=50)

# endregion

# region Recursive chunking

# Intenta respetar separadores jerarquicos antes de cortar por caracteres,
# buscando fragmentos mas coherentes para una consulta RAG.
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

recursive_chunks = splitter.split_text(pdf_complete_text)

# endregion

# region Section-based chunking

# Depende de que los encabezados sigan este patron; si el formato cambia, la
# segmentacion puede producir secciones incorrectas.
pattern = r"(?=\d+\.\s[A-Z])"

section_chunks = re.split(pattern, pdf_complete_text)

# endregion

# region Semantic chunking

# Esta estrategia usa embeddings para detectar cambios de significado. Es mas
# costosa que cortar por caracteres, pero puede respetar mejor las ideas.
from langchain_experimental.text_splitter import SemanticChunker  # type: ignore

from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = SemanticChunker(embeddings)

semantic_chunks = splitter.create_documents([pdf_complete_text])

semantic_chunks = [
    doc.page_content
    for doc in splitter.create_documents([pdf_complete_text])
]

# endregion

strategies = {
    "Fixed-size": fixed_chunks,
    "Semantic": semantic_chunks,
    "Recursivo": recursive_chunks,
    "Sliding window": sliding_window_chunks,
    "Por secciones": section_chunks,
}

for name, chunks in strategies.items():

    print(f"\n{name}")

    # Cantidad y longitud ayudan a comparar costos, pero no bastan para medir
    # calidad: tambien habria que evaluar precision y cobertura del retrieval.
    print(f"Chunks: {len(chunks)}")

    print("Lengths:")

    print([len(c) for c in chunks[:10]])