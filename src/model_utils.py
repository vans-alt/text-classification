from google.colab import drive
drive.mount('/content/drive')

from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/models/distilbert-base-uncased",
    num_labels=2  # Binary classification: 0 = negative, 1 = positive
)

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
