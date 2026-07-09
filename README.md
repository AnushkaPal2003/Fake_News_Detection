# Fake News Detection | RNN · LSTM · GRU · Word2Vec · MLflow · Streamlit

Detects fake news by comparing 4 models — Simple RNN, LSTM, GRU, and a TF-IDF + Logistic Regression baseline — trained on the ISOT Fake News Dataset.

## Why This Project
Fake news is a real-world NLP problem. This project demonstrates how sequence models (RNN, LSTM, GRU) learn contextual patterns in text using Word2Vec embeddings, and how they compare against a classical TF-IDF baseline — a common interview question in NLP roles.

## Models Compared
| Model | Type | Key Strength |
|---|---|---|
| Simple RNN | Sequence | Fast, struggles with long sequences |
| LSTM | Sequence | Handles long-range dependencies via gates |
| GRU | Sequence | Faster than LSTM, similar accuracy |
| TF-IDF + LR | Classical | Strong baseline, interpretable |

## Tech Stack
- **NLP**: Word2Vec (Gensim), TF-IDF
- **Deep Learning**: TensorFlow/Keras — RNN, LSTM, GRU
- **Experiment Tracking**: MLflow
- **Deployment**: Streamlit
- **CI/CD**: GitHub Actions

## Project Structure
```
fake-news-detection/
├── data/                    # ISOT dataset CSVs (True.csv, Fake.csv)
├── src/
│   ├── preprocess.py        # data cleaning, tokenization, padding
│   ├── embeddings.py        # Word2Vec training, embedding matrix
│   ├── models.py            # RNN, LSTM, GRU model definitions
│   ├── train.py             # training loop with MLflow logging
│   └── evaluate.py          # metrics, confusion matrix, comparison chart
├── app.py                   # Streamlit UI
├── .github/workflows/ci.yml # GitHub Actions CI pipeline
└── requirements.txt
```

## Setup

### 1. Download Dataset
Download from Kaggle: [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `True.csv` and `Fake.csv` in the `data/` folder.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
```bash
python -m src.train
```
This will:
- Preprocess and tokenize text
- Train Word2Vec embeddings on corpus
- Train RNN, LSTM, GRU with early stopping
- Train TF-IDF + LR baseline
- Log all experiments to MLflow
- Save models and results to `artifacts/`

### 4. View MLflow Dashboard
```bash
mlflow ui
```
Open `http://localhost:5000` to compare experiments.

### 5. Run Streamlit App
```bash
streamlit run app.py
```

## Key Design Decisions
- **Word2Vec over random embeddings**: trained on the actual news corpus for domain-specific representations
- **Frozen embeddings**: Word2Vec weights kept frozen during training to preserve learned representations
- **Early stopping**: prevents overfitting with patience=3
- **4-model comparison**: helps understand trade-offs between sequence models and classical approaches
