📦 Text Classification with Transformers

🧠 Overview

This project implements a binary text classification pipeline using Hugging Face Transformers and the IMDB dataset. The goal is to fine-tune a pre-trained transformer (DistilBERT) to classify movie reviews as **positive** or **negative**.

<pre>text-classification/
├── notebooks/
│   └── text_classification_pipeline.ipynb      # End-to-end pipeline in Jupyter
│
├── src/
│   ├── config.py                               # Hyperparameters & model paths
│   ├── data_preprocessing.py                   # Data cleaning, tokenization
│   ├── model_utils.py                          # Load/save model, pipeline logic
│   ├── train_model.py                          # Training using HuggingFace Trainer
│   └── __init__.py
│
├── models/
│   ├── trained/                                # Final fine-tuned model
│   ├── tokenizer/                              # Tokenizer files
│   └── checkpoints/                            # Intermediate checkpoints
│
├── reports/
│   ├── submission.md                           # Approach, learnings, future work
│   ├── execution_report.md                     # CLI steps + screenshots
│   ├── evaluation_metrics.json                 # Accuracy, Precision, Recall, F1
│   └── training_curve.png                      # Training vs. Eval F1 loss plot
│
├── scripts/
│   └── train.py                                # CLI entry point to train the model
│
├── requirements.txt                            # Python dependencies
└── README.md                                    # Project overview & instructions
</pre>

🛠️ Setup

1. Clone this repository:

```bash
git clone <repo-url>
cd text-classification-pipeline
```

2. (Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```


🚀 How to Run

📌 Option 1: From notebook

Run `notebooks/text-classification.ipynb` in Google Colab or Jupyter Notebook for full pipeline.

📌 Option 2: From script

```bash
python train.py
```

📊 Evaluation

| Metric    | Score  |
| --------- | -------|
| Accuracy  | ~0.905 |
| F1 Score  | ~0.906 |
| Precision | ~0.905 |
| Recall    | ~0.905 |

📌 Visualizations available in `reports/train loss, eval f1 plot.png`

🔎 Key Features

* Modular architecture (data, model, config separated)
* Easy experimentation with `Trainer`
* WANDB logging integrated
* Intermediate checkpoints and best model saving
* Ready to extend for multilingual classification (e.g. XLM-Roberta)


✍️ Author

Vanshita Gupta

🪄 License

Open-source for educational purposes. Modify and reuse with attribution.

Happy fine-tuning! 🤗
"# text-classification" 
