import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar y preprocesar los datos
df = pd.read_csv("new-df.csv")

# Carpeta para guardar las Graficas
output_dir = "Data-Classification-Graphs"
os.makedirs(output_dir, exist_ok=True)

# Convertir release_date a tipo datetime y extraer el año
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Manejar valores faltantes eliminando filas con NaN en columnas relevantes
df.dropna(subset=['popularity', 'vote_count', 'release_year'], inplace=True)

# Codificar 'original_language' (categórica) a valores numéricos
df['original_language'] = df['original_language'].astype('category').cat.codes

# Crear una columna binaria para la popularidad (1: Alta, 0: Baja) según la mediana
df['popularity_binary'] = (df['popularity'] > df['popularity'].median()).astype(int)

# Seleccionar características y objetivo
X = df[['vote_count', 'release_year', 'original_language']]  # Características
y = df['popularity_binary']  # Objetivo

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo
k = 5  # Número de vecinos
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predecir la popularidad
y_pred = knn.predict(X_test)

# Evaluar el modelo
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("\nPrecisión del Modelo:", accuracy_score(y_test, y_pred))

# Crear y guardar una gráfica de la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Baja', 'Alta'], yticklabels=['Baja', 'Alta'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
output_path = os.path.join(output_dir, "Grafica-KNN-1.png")
plt.savefig(output_path)
print(f"Gráfica de la matriz de confusión guardada en: {output_path}")
plt.show()

# Crear y guardar una gráfica de precisión vs número de vecinos
k_values = range(1, 21)  # Probar valores de k desde 1 hasta 20
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Graficar precisión vs número de vecinos
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Precisión')
plt.title('Precisión vs Número de Vecinos (k)')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Precisión')
plt.xticks(k_values)
plt.grid(alpha=0.5)
plt.legend()

# Guardar la gráfica
output_path_accuracy = "Data-Classification-Graphs/precision_vs_k.png"
plt.savefig(output_path_accuracy)
print(f"Gráfica de precisión vs k guardada en: {output_path_accuracy}")
plt.show()