📄 Submission: Text Classification Pipeline (ML/NLP Engineer Intern)

🔍 Problem Statement

Build a transformer-based binary text classification pipeline using the IMDB sentiment dataset. Fine-tune a pre-trained Hugging Face model (DistilBERT) and evaluate using F1-score, precision, recall, and accuracy.

🚀 Approach Summary

1. Dataset

* Used **IMDB Movie Review Dataset** containing 50,000 labeled reviews.
* Each review is labeled as **positive** or **negative**.

2. Data Preprocessing

* Mapped "positive" → 1, "negative" → 0.
* Used Hugging Face `Datasets` to convert pandas to `Dataset` format.
* Tokenized with `DistilBertTokenizerFast` with truncation and padding.

3. Model Selection

* Used **DistilBERT**: a lightweight version of BERT, good trade-off between performance and training time.
* Loaded using `DistilBertForSequenceClassification` with `num_labels=2`.

4. Training Setup

* Training with Hugging Face `Trainer` class.
* Key training arguments:

  * `learning_rate`: 2e-5
  * `epochs`: 3
  * `batch_size`: 16
  * `load_best_model_at_end`: True
* Checkpoints saved at steps: 500, 1000, 1500
* Experiment tracking done using **Weights & Biases (wandb)**.

5. Evaluation

* Metrics reported: **accuracy**, **F1-score**, **precision**, and **recall**.
* Evaluation done on validation split.
* Visualized results using `confusion_matrix.png` and saved metrics to `evaluation_metrics.json`.

---

## 📊 Results Snapshot

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | ~0.905 |
| F1 Score  | ~0.906 |
| Precision | ~0.906 |
| Recall    | ~0.905 |

 Key Learnings

* Hugging Face's `Trainer` API significantly simplifies training workflows for NLP tasks.
* Proper tokenization and label mapping are critical for clean training.
* WandB is highly effective for monitoring training and evaluating experiments.
* DistilBERT performs well for text classification with fast training.
* Organizing code modularly in `src/` helps scale or swap components easily.

🎯 Bonus Scope (Multilingual)

To extend this to multilingual sentiment analysis:

* Use `xlm-roberta-base` or `bert-base-multilingual-cased` as the pre-trained model.
* Replace the dataset with one like **Amazon Multilingual Reviews**.
* Adjust tokenizer and model accordingly.



