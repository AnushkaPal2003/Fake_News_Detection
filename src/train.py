import numpy as np
import mlflow
import mlflow.keras
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle

from src.preprocess import preprocess
from src.embeddings import train_word2vec, build_embedding_matrix
from src.models import get_model
from src.evaluate import get_metrics

EPOCHS = 10
BATCH_SIZE = 64
MODELS_TO_TRAIN = ["rnn", "lstm", "gru"]


def train_deep_models(X_train, X_test, y_train, y_test, embedding_matrix):
    results = {}

    for model_name in MODELS_TO_TRAIN:
        print(f"\n--- Training {model_name.upper()} ---")

        with mlflow.start_run(run_name=model_name.upper()):
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("batch_size", BATCH_SIZE)

            model = get_model(model_name, embedding_matrix)

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=1
            )

            # evaluate
            metrics = get_metrics(model, X_test, y_test, model_type="deep")

            # log metrics to mlflow
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])

            # save model
            os.makedirs("artifacts/models", exist_ok=True)
            model.save(f"artifacts/models/{model_name}_model.h5")
            mlflow.keras.log_model(model, model_name)

            results[model_name] = metrics
            print(f"{model_name.upper()} — Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    return results


def train_baseline(X_train_raw, X_test_raw, y_train, y_test):
    # TF-IDF + Logistic Regression baseline
    # X_train_raw/X_test_raw/y_train/y_test all come from the SAME split
    # produced by preprocess() — no separate reload/resplit here, so row
    # counts always match.
    print("\n--- Training Baseline (TF-IDF + Logistic Regression) ---")

    with mlflow.start_run(run_name="TF-IDF_LR_Baseline"):
        mlflow.log_param("model_type", "tfidf_lr")
        mlflow.log_param("max_features", 20000)

        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train_raw)
        X_test_tfidf = tfidf.transform(X_test_raw)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_tfidf, y_train)

        y_pred = lr.predict(X_test_tfidf)
        y_prob = lr.predict_proba(X_test_tfidf)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
        }

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])

        # save artifacts
        with open("artifacts/tfidf.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        with open("artifacts/lr_model.pkl", "wb") as f:
            pickle.dump(lr, f)

        print(f"Baseline — Accuracy: {metrics['accuracy']} | F1: {metrics['f1_score']} | ROC-AUC: {metrics['roc_auc']}")

    return metrics


def run_training(data_dir="data"):
    mlflow.set_experiment("fake-news-detection")

    # preprocess — now returns raw text splits (X_train_raw, X_test_raw)
    # alongside the padded sequences, all from the same split/filter, so
    # nothing downstream needs to reload or resplit the data.
    print("Preprocessing data...")
    X_train_pad, X_test_pad, y_train, y_test, tokenizer, X_train_raw, X_test_raw = preprocess(data_dir)

    # train Word2Vec on the training text only (avoids leaking test text
    # into the embeddings)
    print("\nTraining Word2Vec embeddings...")
    w2v_model = train_word2vec(X_train_raw)
    embedding_matrix = build_embedding_matrix(tokenizer, w2v_model)

    # train deep models
    deep_results = train_deep_models(X_train_pad, X_test_pad, y_train, y_test, embedding_matrix)

    # train baseline — reuses the exact same split, no mismatch possible
    baseline_results = train_baseline(X_train_raw, X_test_raw, y_train, y_test)

    # combine all results
    all_results = {**deep_results, "tfidf_lr": baseline_results}

    # save results summary
    import json
    with open("artifacts/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Final Results ===")
    for model, metrics in all_results.items():
        print(f"{model.upper()}: Accuracy={metrics['accuracy']} | F1={metrics['f1_score']} | ROC-AUC={metrics['roc_auc']}")

    return all_results


if __name__ == "__main__":
    run_training()