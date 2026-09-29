import pymupdf

pdf_path = "attention-is-all-you-need.pdf"

# Esta es la etapa de ingesta de una knowledge base: primero se transforma el
# documento original en texto que luego podra limpiarse, dividirse y vectorizarse.
doc = pymupdf.open(pdf_path)

print(f"El documento tiene {len(doc)} páginas\n")

# Se inspeccionan solo dos paginas para observar el resultado de la extraccion;
# esto no representa una carga completa del PDF.
for numero_pagina, pagina in enumerate(doc[:2], start=1):
    print("=" * 60)
    print(f"Página {numero_pagina}")
    print("=" * 60)

    # La extraccion lineal puede perder tablas, columnas, formato y jerarquia
    # visual. Esa perdida debe considerarse antes de alimentar un RAG.
    texto = pagina.get_text()

    print(texto)
    print("\n")

# Liberar el recurso evita mantener abierto el archivo despues de la ingesta.
doc.close()