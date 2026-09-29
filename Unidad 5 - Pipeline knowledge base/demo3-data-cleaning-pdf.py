import pymupdf
import re

pdf_path = "attention-is-all-you-need.pdf"

doc = pymupdf.open(pdf_path)

print(f"El documento tiene {len(doc)} páginas\n")


def clean_text(text: str) -> str:
    # La limpieza prepara el texto extraido para chunking y embeddings:
    # reduce ruido de formato sin cambiar deliberadamente su significado.
    # La calidad de esta etapa afecta directamente a la recuperacion semantica.
    text = re.sub(r"-\n", "", text)

    # El PDF puede insertar saltos de linea por maquetacion, no por cambio de
    # idea. Unificarlos facilita que un chunk contenga frases completas.
    text = re.sub(r"\n+", " ", text)

    # Normalizar espacios evita diferencias superficiales entre fragmentos.
    text = re.sub(r"\s+", " ", text)

    # Normalizar Unicode reduce variantes tipograficas que pueden perjudicar
    # busquedas semanticas o comparaciones.
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
