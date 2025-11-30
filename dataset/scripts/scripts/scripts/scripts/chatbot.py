from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te

tokenizer = AutoTokenizer.from_pretrained("modelo/bert_entrenado")
model = AutoModelForSequenceClassification.from_pretrained("modelo/bert_entrenado")

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

respuestas = {
    "inscripcion": "Para inscribirte debes ingresar a la página del ITSE y llenar el formulario.",
    "requisitos": "Los requisitos principales son: cédula, diploma y fotos tamaño carnet.",
    "becas": "Las becas disponibles incluyen beca socioeconómica y beca de excelencia.",
    "horarios": "Los horarios dependen del técnico.",
    "contacto": "Puedes contactar al ITSE al correo info@itse.ac.pa"
}

vader = SentimentIntensityAnalyzer()

print("Chatbot ITSE — escribe 'salir' para terminar")

while True:
    texto = input("Tú: ")
    if texto.lower() == "salir":
        break

    pred = pipe(texto)[0]["label"]
    print("Bot:", respuestas.get(pred, "No entendí tu consulta."))

    print("Sentimiento:", vader.polarity_scores(texto))
    print("Emociones:", te.get_emotion(texto))
