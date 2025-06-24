📦 Text Classification with Transformers

🧠 Overview

This project implements a binary text classification pipeline using Hugging Face Transformers and the IMDB dataset. The goal is to fine-tune a pre-trained transformer (DistilBERT) to classify movie reviews as **positive** or **negative**.

📁 Project Structure

├── notebooks/
│   ├── text-classification.ipynb       # Full pipeline execution notebook   
│
├── src/
│   ├── train_model.py                  # Model training using Hugging Face Trainer
│   ├── data_preprocessing.py           # Text cleaning, tokenization, dataset prep
│   ├── model_utils.py                  # Model loading/saving and pipeline use
│   └── config.py                       # All hyperparameters and model paths
│
├── models/
│   ├── trained_model/                  # Fine-tuned model weights
│   ├── tokenizer/                      # Tokenizer files
│   └── checkpoints/                    # Intermediate training checkpoints
│
├── reports/
│   ├── submission.md                   # Full write-up of approach & learnings
│   ├── execution_report.md             # Execution steps + screenshots
│   ├── evaluation_metrics.json         # Accuracy, F1, Precision, Recall
│   └── training_curve.png              # Graph of training loss vs. eval F1
│
├── train.py                            # Entry point to run full training
├── requirements.txt                    # Python dependencies
└── README.md                           # This file


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
