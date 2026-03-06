import joblib
import numpy as np
import requests
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------- LOAD V1 MODEL --------
v1_model = joblib.load("models/v1/sentiment_model_v1.pkl")


def predict_v1(text):
    prediction = v1_model.predict([text])[0]
    return prediction


# -------- LOAD TOKENIZER --------
tokenizer = joblib.load("models/v2/tokenizer_v2.pkl")

# -------- LOAD LABEL ENCODER --------
label_encoder = joblib.load("models/v2/label_encoder_v2.pkl")


# -------- TF SERVING URL --------
TF_SERVING_URL = os.getenv(
    "TF_SERVING_URL",
    "http://tfserving:8501/v1/models/sentiment_v2:predict",
)
def predict_v2(text):

    # convert text → sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Must match model training input length (max_len=100 in train_v2.py)
    padded = pad_sequences(sequence, maxlen=100, padding="post")

    data = {
        "instances": padded.tolist()
    }

    response = requests.post(TF_SERVING_URL, json=data)

    # print tensorflow response for debugging
    print("TF RESPONSE:", response.text)

    if response.status_code != 200:
        raise Exception(response.text)

    prediction = np.array(response.json()["predictions"])

    predicted_class = np.argmax(prediction, axis=1)

    label = label_encoder.inverse_transform(predicted_class)

    return label[0]
