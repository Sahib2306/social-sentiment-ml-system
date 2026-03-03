import joblib
import requests
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===== V1 MODEL =====
v1_model = joblib.load("models/v1/sentiment_model_v1.pkl")

def predict_v1(text: str):
    return v1_model.predict([text])[0]


# ===== V2 CONFIG =====
tokenizer = joblib.load("models/v2/tokenizer_v2.pkl")
label_encoder = joblib.load("models/v2/label_encoder_v2.pkl")

MAX_LEN = 100
TF_SERVING_URL = "http://tfserving:8501/v1/models/sentiment_v2:predict"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_v2(text: str):
    # Clean
    text = clean_text(text)

    # Tokenize
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

    # Prepare request
    data = {
        "instances": padded.tolist()
    }

    response = requests.post(TF_SERVING_URL, json=data)
    prediction = response.json()["predictions"][0]

    predicted_index = np.argmax(prediction)
    label = label_encoder.inverse_transform([predicted_index])[0]

    return label