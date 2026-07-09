import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# config
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 300
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(data_dir="data"):
    # load true and fake news CSVs from kaggle ISOT dataset
    true_df = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake_df = pd.read_csv(os.path.join(data_dir, "Fake.csv"))

    # add labels: 1 = real, 0 = fake
    true_df["label"] = 1
    fake_df["label"] = 0

    # combine and shuffle
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df


def clean_text(text):
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # remove special characters and digits
    text = re.sub(r"[^a-z\s]", "", text)
    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(data_dir="data"):
    df = load_data(data_dir)

    # combine title + text for richer input
    df["content"] = df["title"] + " " + df["text"]
    df["content"] = df["content"].apply(clean_text)

    # remove empty rows — this is the ONLY filtering step, applied once,
    # so every downstream split (deep models and baseline) sees the same
    # row count and the same train/test split.
    df = df[df["content"].str.len() > 10].reset_index(drop=True)

    # split
    X = df["content"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # tokenize
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # pad sequences to same length
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")

    # save tokenizer for later use in app
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Train size: {len(X_train_pad)}, Test size: {len(X_test_pad)}")
    print(f"Vocab size: {len(tokenizer.word_index)}")

    # Return raw (untokenized) text splits too — X_train/X_test here are the
    # raw cleaned strings before tokenization, from the SAME split used for
    # the padded sequences above. The baseline (TF-IDF+LR) should reuse these
    # directly instead of re-loading and re-splitting the data, which is what
    # caused the row-count mismatch before.
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, X_train, X_test