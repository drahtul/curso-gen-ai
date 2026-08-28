import pymupdf

pdf_path = "attention-is-all-you-need.pdf"

doc = pymupdf.open(pdf_path)

print(f"El documento tiene {len(doc)} páginas\n")

for numero_pagina, pagina in enumerate(doc[:2], start=1):
    print("=" * 60)
    print(f"Página {numero_pagina}")
    print("=" * 60)

    texto = pagina.get_text()

    print(texto)
    print("\n")

doc.close()