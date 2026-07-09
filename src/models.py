from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 300
EMBEDDING_DIM = 100


def build_rnn(embedding_matrix):
    # simple RNN — fastest but weakest at long sequences
    model = Sequential([
        Embedding(
            input_dim=MAX_VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQ_LENGTH,
            trainable=False  # freeze Word2Vec weights
        ),
        SpatialDropout1D(0.2),
        SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(embedding_matrix):
    # LSTM - handles long-range dependencies better than RNN
    model = Sequential([
        Embedding(
            input_dim=MAX_VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQ_LENGTH,
            trainable=False
        ),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_gru(embedding_matrix):
    # GRU - faster than LSTM, similar performance
    model = Sequential([
        Embedding(
            input_dim=MAX_VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQ_LENGTH,
            trainable=False
        ),
        SpatialDropout1D(0.2),
        GRU(64, dropout=0.2),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_model(model_name, embedding_matrix):
    models = {
        "rnn": build_rnn,
        "lstm": build_lstm,
        "gru": build_gru
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name](embedding_matrix)
