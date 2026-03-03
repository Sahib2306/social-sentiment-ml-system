# 🚀 Social Media Sentiment Analysis – Production ML System

A full-stack, production-style Machine Learning system that compares:

- **V1** → Traditional ML (TF-IDF + Logistic Regression)
- **V2** → Deep Learning (LSTM deployed using TensorFlow Serving)

The system is fully containerized using **Docker** and orchestrated with **Docker Compose**.

---

# 🏗 Architecture
User
↓
FastAPI (Container 1)
↓
TensorFlow Serving (Container 2)
↓
LSTM Model (V2)

- V1 runs directly inside FastAPI
- V2 runs inside TensorFlow Serving
- FastAPI handles preprocessing & postprocessing
- Docker manages multi-container communication

---

# 📁 Project Structure
Social_Sentiment_App/
│
├── app/
│   ├── main.py
│   ├── utils.py
│   └── templates/
│       └── index.html
│
├── models/
│   ├── v1/
│   │   ├── train_v1.py
│   │   └── sentiment_model_v1.pkl
│   │
│   └── v2/
│       ├── train_v2.py
│       ├── tokenizer_v2.pkl
│       ├── label_encoder_v2.pkl
│       └── serving/
│           └── sentiment_v2/
│               └── 1/
│                   ├── saved_model.pb
│                   ├── variables/
│                   └── assets/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

---

# 🧠 Models

## 🔹 V1 – Traditional ML
- Text cleaning
- TF-IDF vectorization
- Logistic Regression classifier
- Saved using `joblib`

## 🔹 V2 – Deep Learning
- Text tokenization
- Padding sequences
- Embedding + LSTM
- Softmax classification
- Exported in TensorFlow SavedModel format
- Deployed using TensorFlow Serving

---

# 🛠 Tech Stack

- Python 3.11
- FastAPI
- TensorFlow 2.20
- Scikit-learn
- Docker
- Docker Compose
- TensorFlow Serving
- HTML + Jinja2

---

# ⚙️ How to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd Social_Sentiment_App

2️⃣ Build & Run with Docker
docker-compose up --build

3️⃣ Open in Browser

http://localhost:8000

🔍 API Endpoints

Web Interface
GET  /
POST /predict

TensorFlow Serving REST

http://localhost:8501/v1/models/sentiment_v2


🐳 Docker Setup

The system runs two containers:

FastAPI Container
	•	Handles:
	•	Text cleaning
	•	Tokenization
	•	Padding
	•	Label decoding

TensorFlow Serving Container
	•	Serves:
	•	LSTM model
	•	REST inference endpoint

⸻

📊 Current Model Performance
	•	V1 performs reasonably well using traditional ML.
	•	V2 infrastructure is production-ready but requires further tuning for higher accuracy.

⸻

🚀 Future Improvements
	•	Improve LSTM accuracy
	•	Add Bidirectional LSTM
	•	Add EarlyStopping
	•	Add logging & monitoring
	•	Deploy to cloud (AWS / GCP / Render)
	•	Add Swagger JSON endpoints
	•	CI/CD pipeline

⸻

🎯 Learning Outcomes

This project demonstrates:
	•	End-to-end ML pipeline
	•	Model comparison (Traditional vs Deep Learning)
	•	Model serialization
	•	TensorFlow Serving deployment
	•	Multi-container Docker architecture
	•	Inter-service communication
	•	Production-level ML system design

⸻

👨‍💻 Author

Sahib Chouhan
BCA (AI & ML) Student
Machine Learning & Full Stack Enthusiast
