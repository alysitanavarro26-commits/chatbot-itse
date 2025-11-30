import pandas as pd
import spacy

nlp = spacy.load("es_core_news_sm")

df = pd.read_csv("dataset.csv")

def limpiar(texto):
    texto = texto.lower()
    texto = texto.replace("¿", "").replace("?", "")
    texto = texto.replace("¡", "").replace("!", "")
    return texto

def lematizar(texto):
    doc = nlp(texto)
    return " ".join([token.lemma_ for token in doc])

df["texto_limpio"] = df["texto"].apply(limpiar)
df["texto_lematizado"] = df["texto_limpio"].apply(lematizar)

df.to_csv("dataset_limpio.csv", index=False)

df.head()
