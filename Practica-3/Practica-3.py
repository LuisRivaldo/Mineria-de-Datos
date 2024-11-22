import pandas as pd
import os

# Cargar el CSV
df = pd.read_csv('C:/Users/luiz0/OneDrive/Escritorio/Mineria-de-Datos/new-df.csv')

#Estadisticas descriptivas basicas
descriptive_stats = df.describe()
print(descriptive_stats.round(2))

#distribucion de datos por categorias
for column in df.select_dtypes(include='object'):
    print(f"Distribución de {column}:\n{df[column].value_counts()}\n")

for column in df.columns:
    print(f"Valores únicos en {column}: {df[column].nunique()}")

# Convertir las fechas a formato datetime y procesar los géneros
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df_genres = df['genres'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
df_genres.name = 'genre'
df_with_genres = df.drop(columns=['genres']).join(df_genres)

#Definir carpeta para guardar los archivos csv
output_folder = 'C:/Users/luiz0/OneDrive/Escritorio/Mineria-de-Datos/Descriptive_Statistics_CSVs/'

#Crear carpeta si no existe
os.makedirs(output_folder, exist_ok=True)

# 1. Distribución de Popularidad
popularity_distribution = df['popularity'].value_counts().reset_index()
popularity_distribution.columns = ['Popularidad', 'Frecuencia']
popularity_distribution.to_csv(os.path.join(output_folder, 'distribucion_popularidad.csv'), index=False)

# 2. Popularidad Promedio por Idioma
popularity_by_language = df.groupby('original_language')['popularity'].mean().reset_index()
popularity_by_language.columns = ['Idioma', 'Popularidad Promedio']
popularity_by_language.to_csv(os.path.join(output_folder, 'popularidad_promedio_por_idioma.csv'), index=False)

# 3. Relación entre Popularidad y Número de Votos
popularity_votes_relation = df[['popularity', 'vote_count']]
popularity_votes_relation.to_csv(os.path.join(output_folder, 'relacion_popularidad_numero_votos.csv'), index=False)

# 4. Películas Más Comunes por Género
genre_counts = df_with_genres['genre'].value_counts().reset_index()
genre_counts.columns = ['Género', 'Cantidad de Películas']
genre_counts.to_csv(os.path.join(output_folder, 'peliculas_por_genero.csv'), index=False)

# 5. Tendencia de Popularidad a lo Largo del Tiempo
popularity_trend = df.groupby('release_year')['popularity'].mean().reset_index()
popularity_trend.columns = ['Año', 'Popularidad Promedio']
popularity_trend.to_csv(os.path.join(output_folder, 'tendencia_popularidad_año.csv'), index=False)