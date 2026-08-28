import pymupdf
import re

pdf_path = "attention-is-all-you-need.pdf"

def clean_text(text: str) -> str:
    # Remove hyphenation across lines
    text = re.sub(r"-\n", "", text)

    # Replace line breaks with spaces
    text = re.sub(r"\n+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Normalize Unicode punctuation
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

for page in doc:
    pdf_complete_text += clean_text(page.get_text())
    pdf_complete_text += "\n\n"

# region Fixed-size chunking

def fixed_size_chunking(text, chunk_size=600):
    return [
        text[i:i+chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

fixed_chunks = fixed_size_chunking(pdf_complete_text, 600)

# endregion

# region Sliding window chunking

def sliding_window_chunking(text, chunk_size=500, overlap=100):

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

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

recursive_chunks = splitter.split_text(pdf_complete_text)

# endregion

# region Section-based chunking

pattern = r"(?=\d+\.\s[A-Z])"

section_chunks = re.split(pattern, pdf_complete_text)

# endregion

# region Semantic chunking

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

    print(f"Chunks: {len(chunks)}")

    print("Lengths:")

    print([len(c) for c in chunks[:10]])