import streamlit as st
import numpy as np
import pickle
import json
import os
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LENGTH = 300

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection")
st.markdown("Detects fake news using RNN, LSTM, GRU, and TF-IDF baseline models trained on the ISOT dataset.")


@st.cache_resource
def load_artifacts():
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("artifacts/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("artifacts/lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    rnn = load_model("artifacts/models/rnn_model.h5")
    lstm = load_model("artifacts/models/lstm_model.h5")
    gru = load_model("artifacts/models/gru_model.h5")
    return tokenizer, tfidf, lr_model, rnn, lstm, gru


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_deep(model, text, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Real" if prob >= 0.5 else "Fake"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, float(confidence)


def predict_baseline(text, tfidf, lr_model):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    prob = lr_model.predict_proba(vec)[0][1]
    label = "Real" if prob >= 0.5 else "Fake"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, float(confidence)


# load models
if os.path.exists("artifacts/tokenizer.pkl"):
    tokenizer, tfidf, lr_model, rnn, lstm, gru = load_artifacts()

    # input
    st.subheader("Enter News Article")
    user_input = st.text_area("Paste news title + content here:", height=150,
                               placeholder="Enter the news article you want to check...")

    if st.button("Detect", type="primary") and user_input.strip():
        st.subheader("Results")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            label, conf = predict_deep(rnn, user_input, tokenizer)
            color = "🟢" if label == "Real" else "🔴"
            st.metric("RNN", f"{color} {label}", f"{conf:.1%} confidence")

        with col2:
            label, conf = predict_deep(lstm, user_input, tokenizer)
            color = "🟢" if label == "Real" else "🔴"
            st.metric("LSTM", f"{color} {label}", f"{conf:.1%} confidence")

        with col3:
            label, conf = predict_deep(gru, user_input, tokenizer)
            color = "🟢" if label == "Real" else "🔴"
            st.metric("GRU", f"{color} {label}", f"{conf:.1%} confidence")

        with col4:
            label, conf = predict_baseline(user_input, tfidf, lr_model)
            color = "🟢" if label == "Real" else "🔴"
            st.metric("TF-IDF + LR", f"{color} {label}", f"{conf:.1%} confidence")

    # model comparison chart
    st.divider()
    st.subheader("Model Performance Comparison")

    if os.path.exists("artifacts/results.json"):
        with open("artifacts/results.json") as f:
            results = json.load(f)

        import pandas as pd
        df = pd.DataFrame(results).T.reset_index()
        df.columns = ["Model", "Accuracy", "F1 Score", "ROC-AUC"]
        df["Model"] = df["Model"].str.upper()
        st.dataframe(df.style.highlight_max(subset=["Accuracy", "F1 Score", "ROC-AUC"],
                                             color="lightgreen"), use_container_width=True)

    if os.path.exists("artifacts/model_comparison.png"):
        st.image("artifacts/model_comparison.png", use_column_width=True)

else:
    st.warning("Models not found. Please run `python -m src.train` first to train the models.")
    st.code("python -m src.train", language="bash")
