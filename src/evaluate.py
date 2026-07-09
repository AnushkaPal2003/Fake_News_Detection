import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_metrics(model, X_test, y_test, model_type="deep"):
    if model_type == "deep":
        y_prob = model.predict(X_test).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4)
    }


def plot_confusion_matrix(y_test, y_pred, model_name, save_dir="artifacts"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"],
                yticklabels=["Fake", "Real"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/cm_{model_name}.png", bbox_inches="tight")
    plt.close()


def plot_model_comparison(results, save_dir="artifacts"):
    models = list(results.keys())
    accuracy = [results[m]["accuracy"] for m in models]
    f1 = [results[m]["f1_score"] for m in models]
    roc_auc = [results[m]["roc_auc"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, accuracy, width, label="Accuracy")
    ax.bar(x, f1, width, label="F1 Score")
    ax.bar(x + width, roc_auc, width, label="ROC-AUC")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Fake News Detection")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.set_ylim(0.5, 1.05)

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/model_comparison.png", bbox_inches="tight")
    plt.close()
    print("Model comparison chart saved.")
