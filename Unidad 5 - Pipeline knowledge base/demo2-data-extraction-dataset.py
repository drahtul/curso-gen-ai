import pandas as pd

csv_path = "imdb-top-1000.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} movies.\n")

print(df.head())


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


# Crear documentos a partir del data set

documents = []

for _, movie in df.iterrows():
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
Overview:
{movie['Overview']}
""".strip()

    documents.append(document)

print(f"Created {len(documents)} documents.\n")

print(documents[0])