import pandas as pd
import re

def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü¿?0-9\s]", "", texto)
    return texto

df = pd.read_csv("dataset/dataset.csv")
df["text"] = df["text"].apply(limpiar)
df.to_csv("dataset/dataset_limpio.csv", index=False)

print("Listo: dataset_limpio.csv generado")
