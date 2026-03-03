import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("../../data/social_media_sentiment_train.csv")

# 2. Basic text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

# 3. Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# 5. Train
pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipeline.predict(X_val)
print(classification_report(y_val, y_pred))

# 7. Save model
joblib.dump(pipeline, "sentiment_model_v1.pkl")

print("Model V1 saved successfully!")