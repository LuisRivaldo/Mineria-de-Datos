import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar el archivo CSV
df = pd.read_csv("new-df.csv")

# Carpeta para guardar las Graficas
output_dir = "Linear-Models-Graphs"
os.makedirs(output_dir, exist_ok=True)

# Preprocesar datos: eliminar filas con valores nulos y convertir fechas
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['vote_count', 'popularity'])  # Asegurarse de que no haya nulos en las columnas relevantes

# Variables para el modelo lineal
X = df[['vote_count']]  # Variable independiente
y = df['popularity']    # Variable dependiente

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Calcular R2
r2 = r2_score(y_test, y_pred)
print(f"Puntuación R2 del modelo: {r2}")

# Coeficientes del modelo
print(f"Coeficiente: {model.coef_[0]}")
print(f"Intersección: {model.intercept_}")

# Graficar los datos y la regresión
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_count', y='popularity', data=df, alpha=0.5, label='Datos')
plt.plot(X_test, y_pred, color='red', label='Modelo Lineal', linewidth=2)
plt.xlabel("Número de votos")
plt.ylabel("Popularidad")
plt.title("Modelo Lineal: Popularidad vs Número de Votos")
plt.legend()
plt.grid(True)

output_path = os.path.join(output_dir, "modelo_lineal.png")
plt.savefig(output_path, dpi=300)
plt.close()

# Correlación entre las variables
correlation = df['vote_count'].corr(df['popularity'])
print(f"Correlación entre 'vote_count' y 'popularity': {correlation}")
print('\n')

# ---2do Modelo---
# Filtrar y preparar datos
df['release_year'] = df['release_date'].dt.year
df.dropna(subset=['popularity', 'release_year'], inplace=True)  # Eliminar valores faltantes
X = df[['release_year']]  # Variable independiente: año de lanzamiento
y = df['popularity']  # Variable dependiente: popularidad

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular el R2
r2_score = model.score(X_test, y_test)
print(f"R2 Score: {r2_score}")

# Graficar el modelo
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['release_year'], y=y_test, label='Datos reales', alpha=0.5)
plt.plot(X_test['release_year'], y_pred, color='red', label='Modelo Lineal', linewidth=2)
plt.xlabel("Año de Lanzamiento")
plt.ylabel("Popularidad")
plt.title("Relación entre Popularidad y Año de Lanzamiento")
plt.legend()
plt.grid(True)

# Guardar la gráfica
output_path = os.path.join(output_dir, "modelo_lineal_2.png")
plt.savefig(output_path, dpi=300)
plt.close()
print('\n')

# ---3er Modelo---
# Crear una columna para contar el número de géneros
df['num_genres'] = df['genres'].apply(lambda x: len(x.split(',')))

# Filtrar y preparar datos
df.dropna(subset=['popularity', 'num_genres'], inplace=True)  # Eliminar valores faltantes
X = df[['num_genres']]  # Variable independiente: número de géneros
y = df['popularity']  # Variable dependiente: popularidad

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular el R2
r2_score = model.score(X_test, y_test)
print(f"R2 Score: {r2_score}")

# Graficar el modelo
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['num_genres'], y=y_test, label='Datos reales', alpha=0.5)
plt.plot(X_test['num_genres'], y_pred, color='red', label='Modelo Lineal', linewidth=2)
plt.xlabel("Número de Géneros")
plt.ylabel("Popularidad")
plt.title("Relación entre Popularidad y Número de Géneros")
plt.legend()
plt.grid(True)

# Guardar la gráfica
output_path = os.path.join(output_dir, "modelo_lineal_3.png")
plt.savefig(output_path, dpi=300)
plt.close()