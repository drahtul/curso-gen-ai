import pymupdf
import re

pdf_path = "attention-is-all-you-need.pdf"

doc = pymupdf.open(pdf_path)

print(f"El documento tiene {len(doc)} páginas\n")


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


for numero_pagina, pagina in enumerate(doc[:2], start=1):
    print("=" * 60)
    print(f"Página {numero_pagina}")
    print("=" * 60)

    texto = pagina.get_text()

    print(texto)
    print("\n")

    print("-" * 60)
    print(f"Página {numero_pagina} (texto limpio)")
    print("-" * 60)

    texto_limpio = clean_text(texto)

    print(texto_limpio)
    print("\n")

doc.close()
