import pandas as pd

csv_path = "imdb-top-1000.csv"
# El CSV es una fuente estructurada; la lectura conserva sus columnas para
# decidir luego que parte sera contenido semantico y que parte sera metadata.
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} movies.\n")

print(df.head())


# Esta muestra permite revisar columnas, valores faltantes y calidad antes de
# transformar todas las filas en documentos para la knowledge base.
for index, movie in df.head(10).iterrows():
    print("=" * 60)
    print(f"Movie #{index + 1}")
    print("=" * 60)

    print(f"Title: {movie['Series_Title']}")
    print(f"Year: {movie['Released_Year']}")
    print(f"Genre: {movie['Genre']}")
    print(f"Director: {movie['Director']}")
    print(f"IMDb Rating: {movie['IMDB_Rating']}")
    print(f"Overview: {movie['Overview']}")
    print()


# Un embedding trabaja sobre texto. Por eso cada fila tabular se transforma
# en una unidad textual recuperable, aunque los campos tambien podrian
# conservarse por separado como metadata filtrable.
# Crear documentos a partir del data set

documents = []

for _, movie in df.iterrows():
    # Overview aporta significado para la similitud; rating, ano o genero
    # podrian usarse como filtros sin alterar el texto vectorizado.
    document = f"""
Title: {movie['Series_Title']}
Release Year: {movie['Released_Year']}
Certificate: {movie['Certificate']}
Runtime: {movie['Runtime']}
Genre: {movie['Genre']}
IMDb Rating: {movie['IMDB_Rating']}
Meta Score: {movie['Meta_score']}
Director: {movie['Director']}
Stars: {movie['Star1']}, {movie['Star2']}, {movie['Star3']}, {movie['Star4']}
Votes: {movie['No_of_Votes']}
Gross Revenue: {movie['Gross']}
Overview: {movie['Overview']}
""".strip()

    # Cada documento sera una unidad independiente durante el chunking o la
    # indexacion posterior.
    documents.append(document)

print(f"Created {len(documents)} documents.\n")

print(documents[0])