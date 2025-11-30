from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

modelo_base = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(modelo_base)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

dataset = load_dataset("csv", data_files="dataset/dataset_limpio.csv")
dataset = dataset["train"].train_test_split(test_size=0.3)

labels = list(set(dataset["train"]["label"]))
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})
dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    modelo_base,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }

args = TrainingArguments(
    output_dir="modelo/bert_entrenado",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=4
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=metrics
)

trainer.train()
trainer.save_model("modelo/bert_entrenado")
print("Modelo guardado en modelo/bert_entrenado")
