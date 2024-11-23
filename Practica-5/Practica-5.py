import pandas as pd
import numpy as np
from scipy.stats import kruskal

# Cargar el DataFrame (ajustar según tu archivo CSV)
df = pd.read_csv('new-df.csv')

# Asegurar que release_date sea tipo datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# 1. Agrupación por idioma
df['language_group'] = df['original_language']
title_count_language = df.groupby('language_group').size().reset_index(name='num_titles')
print("Agrupación por Idioma (Número de Títulos):")
print(title_count_language)
print("\n")
vote_count_language = df.groupby('original_language')['vote_count'].sum().reset_index(name='total_votes')
print("Agrupación de votos por Idioma:")
print(vote_count_language)
print("\n")

merged_df = pd.merge(title_count_language, vote_count_language, left_on='language_group', right_on='original_language')

# Agrupación por géneros
# Dividir géneros y seleccionar el principal (el primer género)
df['genre_group'] = df['genres'].str.split(',').str[0]
title_count_genre = df.groupby('genre_group').size().reset_index(name='num_titles')
print("Agrupación por Género Principal (Número de Títulos):")
print(title_count_genre)
print("\n")
vote_count_genre = df.groupby('genre_group')['vote_count'].sum().reset_index(name='total_votes')
print("Agrupación de votos por Género Principal:")
print(vote_count_genre)
print("\n")

merged_genre_df = pd.merge(title_count_genre, vote_count_genre, on='genre_group')

# Agrupación por épocas basadas en fechas
# Extraer el año de la fecha
df['release_year'] = df['release_date'].dt.year
# Crear una nueva columna para las décadas
df['epoch_group'] = (df['release_year'] // 10 * 10).astype('Int64') 
df['epoch_group'] = df['epoch_group'].fillna("Desconocido") 

# Agrupación por épocas
title_count_epoch = df.groupby('epoch_group').size().reset_index(name='num_titles')
print("Agrupación por epocas (Número de Títulos):")
print(title_count_epoch)
print("\n")
vote_count_epoch = df.groupby('epoch_group')['vote_count'].sum().reset_index(name='total_votes')
print("Agrupacion de votos por epoca: ")
print(vote_count_epoch)
print("\n")

merged_epoch_df = pd.merge(title_count_epoch, vote_count_epoch, on='epoch_group')

# ---Pruebas Kruskall Wallis---
# 1. Kruskall Wallis con agrupacion por idioma
stat, p_value = kruskal(merged_df['num_titles'], merged_df['total_votes'])

# Mostrar los resultados
print(f"Estadístico H: {stat}")
print(f"Valor p: {p_value}")

# Interpretación de los resultados
alpha = 0.05
if p_value < alpha:
    print("Hay diferencias significativas entre el número de títulos y la suma de votos por idioma.")
else:
    print("No hay diferencias significativas entre el número de títulos y la suma de votos por idioma.")
print("\n")

# 2. Kruskal Wallis con agrupacion por genero
stat, p_value = kruskal(merged_genre_df['num_titles'], merged_genre_df['total_votes'])

# Mostrar los resultados
print(f"Estadístico H: {stat}")
print(f"Valor p: {p_value}")

# Interpretación de los resultados
alpha = 0.05
if p_value < alpha:
    print("Hay diferencias significativas entre el número de títulos y la suma de votos por género principal.")
else:
    print("No hay diferencias significativas entre el número de títulos y la suma de votos por género principal.")
print("\n")

# 3. Kruskal Wallis con agrupacion por epoca
stat, p_value = kruskal(merged_epoch_df['num_titles'], merged_epoch_df['total_votes'])

# Mostrar los resultados
print(f"Estadístico H: {stat}")
print(f"Valor p: {p_value}")

# Interpretación de los resultados
alpha = 0.05
if p_value < alpha:
    print("Hay diferencias significativas entre el número de títulos y la suma de votos por época.")
else:
    print("No hay diferencias significativas entre el número de títulos y la suma de votos por época.")