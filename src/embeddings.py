import numpy as np
from gensim.models import Word2Vec
import os

EMBEDDING_DIM = 100
MAX_VOCAB_SIZE = 20000


def train_word2vec(texts):
    # tokenize into list of word lists for Word2Vec
    tokenized = [text.split() for text in texts]

    # train Word2Vec on corpus
    w2v_model = Word2Vec(
        sentences=tokenized,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )

    os.makedirs("artifacts", exist_ok=True)
    w2v_model.save("artifacts/word2vec.model")
    print(f"Word2Vec trained on {len(tokenized)} documents")

    return w2v_model


def build_embedding_matrix(tokenizer, w2v_model):
    # create embedding matrix for keras embedding layer
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    found = 0
    for word, index in tokenizer.word_index.items():
        if index >= MAX_VOCAB_SIZE:
            continue
        if word in w2v_model.wv:
            embedding_matrix[index] = w2v_model.wv[word]
            found += 1

    print(f"Words found in Word2Vec: {found}/{vocab_size}")
    np.save("artifacts/embedding_matrix.npy", embedding_matrix)

    return embedding_matrix
