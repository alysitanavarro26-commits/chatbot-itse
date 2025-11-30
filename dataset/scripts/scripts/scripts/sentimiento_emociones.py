from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te

vader = SentimentIntensityAnalyzer()

texto = input("Ingresa un mensaje: ")

sentimiento = vader.polarity_scores(texto)
emociones = te.get_emotion(texto)

print("Sentimiento:", sentimiento)
print("Emociones:", emociones)
