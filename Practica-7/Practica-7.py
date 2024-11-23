import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os

# Cargar los datos
df = pd.read_csv("new-df.csv")

# Carpeta para guardar las Graficas
output_dir = "Data-Classification-Graphs"
os.makedirs(output_dir, exist_ok=True)

# Preprocesar los datos
# Asegurarse de que release_date sea tipo datetime y extraer el año
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Manejar valores faltantes (por simplicidad, eliminar filas nulas)
df.dropna(subset=['popularity', 'vote_count', 'release_year', 'original_language'], inplace=True)

# Codificar 'original_language' (categórica) a valores numéricos
df['original_language'] = df['original_language'].astype('category').cat.codes

# Seleccionar características y objetivo
X = df[['popularity', 'vote_count', 'release_year', 'original_language']]  # Características
y = (df['popularity'] > df['popularity'].median()).astype(int)  # Objetivo binario: Popularidad alta o baja

# 3. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Entrenar el modelo
k = 5  # Número de vecinos
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 6. Evaluar el modelo
y_pred = knn.predict(X_test)

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("\nPrecisión del Modelo:", accuracy_score(y_test, y_pred))

# Probar diferentes valores de k y almacenar resultados
k_values = range(1, 21)  # Valores de k desde 1 hasta 20
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Precisión')
plt.title('Precisión vs Número de Vecinos (k)')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Precisión')
plt.xticks(k_values)
plt.grid(alpha=0.5)
plt.legend()

# Guardar la gráfica
output_path = os.path.join(output_dir, "Grafica-KNN.png")
plt.savefig(output_path)
plt.show()
