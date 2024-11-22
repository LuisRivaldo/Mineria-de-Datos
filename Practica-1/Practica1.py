import pandas as pd
from tabulate import tabulate

# Leer el archivo CSV correctamente
df = pd.read_csv('C:/Users/luiz0/OneDrive/Escritorio/Mineria-de-Datos/movie_dataset.csv')
# print(df.head())
# Asegúrate de que 'df' es un DataFrame válido
print(tabulate(df, headers='keys', tablefmt='pretty'))