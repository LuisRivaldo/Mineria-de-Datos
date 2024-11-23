import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Se define la carpeta donde estan los archivos CSV
folder = 'Descriptive_Statistics_CSVs/'

# Archivos CSV creados en la practica 3
csv_files = {
    "Distribución de Popularidad": "distribucion_popularidad.csv",
    "Popularidad Promedio por Idioma": "popularidad_promedio_por_idioma.csv",
    "Relación Popularidad-Votos": "relacion_popularidad_numero_votos.csv",
    "Películas por Género": "peliculas_por_genero.csv",
    "Tendencia de Popularidad por Año": "tendencia_popularidad_año.csv"
}

# Cargar los archivos CSV en DataFrames
dataframes = {key: pd.read_csv(os.path.join(folder, file)) for key, file in csv_files.items()}

#Definir carpeta para guardar los graficos
graphs_folder = 'Data_Visualization_Graphs/'

#Crear carpeta si no existe
os.makedirs(graphs_folder, exist_ok=True)

# Función para generar gráficos
def generate_visualizations():
    # Crear gráficos con cada archivo
    for title, df in dataframes.items():
        plt.figure(figsize=(8, 5))
        
        # Diferentes tipos de gráficos según el tipo de datos
        if "Distribución de Popularidad" in title:
            # Histograma
            sns.histplot(df["Popularidad"], kde=True, color='blue')
            plt.title("Histograma: Distribución de Popularidad")
        
        elif "Popularidad Promedio por Idioma" in title:
            # Gráfico de barras
            sns.barplot(data=df, x="Idioma", y="Popularidad Promedio")
            plt.title("Popularidad Promedio por Idioma")
            plt.xticks(rotation=45)
        
        elif "Relación Popularidad-Votos" in title:
            # Diagrama de dispersión
            sns.scatterplot(data=df, x="vote_count", y="popularity", color='green')
            plt.title("Relación entre Popularidad y Número de Votos")
            plt.xlabel("Número de Votos")
            plt.ylabel("Popularidad")
        
        elif "Películas por Género" in title:
            # Diagrama circular (pie chart)
            plt.pie(df["Cantidad de Películas"], labels=df["Género"], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
            plt.title("Películas por Género")
        
        elif "Tendencia de Popularidad por Año" in title:
            # Gráfico de líneas
            sns.lineplot(data=df, x="Año", y="Popularidad Promedio", marker="o", color='red')
            plt.title("Tendencia de Popularidad a lo Largo del Tiempo")
            plt.xlabel("Año")
            plt.ylabel("Popularidad Promedio")
        
        # Guardar cada gráfico como archivo en la carpeta de gráficos
        plt.tight_layout()
        output_path = os.path.join(graphs_folder, f"{title.replace(' ', '_').lower()}_grafico.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Gráfico guardado: {output_path}")

# Generar visualizaciones
generate_visualizations()