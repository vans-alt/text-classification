import pandas as pd

# Replace 'your_dataset.csv' with the actual uploaded file name
df = pd.read_csv("/content/sample_data/text-classification/data/IMDB Dataset.csv")
# Select the correct columns and rename them
df = df[["review", "sentiment"]]
df = df.rename(columns={"review": "text", "sentiment": "label"})
# Map the sentiment values to integers
df["label"] = df["label"].map({"positive": 1, "negative": 0})
df.head()

from sklearn.model_selection import train_test_split
from datasets import Dataset

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"])

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(pd.DataFrame({"text": train_texts, "label": train_labels}))
val_dataset = Dataset.from_pandas(pd.DataFrame({"text": val_texts, "label": val_labels}))

from transformers import DistilBertTokenizerFast

# Load the tokenizer (from Drive if offline)
tokenizer = DistilBertTokenizerFast.from_pretrained("/content/drive/MyDrive/models/distilbert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

# Apply tokenization
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Rename 'label' to 'labels'
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

# Set format to PyTorch tensors
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
