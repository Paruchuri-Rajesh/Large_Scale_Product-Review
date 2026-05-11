🚨 FraudLens
Large-Scale Product Review Sentiment & Fraud Detection


📌 Overview

FraudLens is an end-to-end big data pipeline that analyzes Amazon product reviews at scale to perform:

💬 Sentiment Classification (Positive / Neutral / Negative)
🕵️ Fraud Detection (Fake / Suspicious Reviews)
⚡ Real-time Streaming Predictions
🤖 LLM-based Fraud Explanations

The system processes 20M+ reviews using distributed computing and serves results via APIs, dashboards, and AI tools.

🏗️ Architecture
⚙️ Tech Stack
Big Data: Apache Spark, Kafka
Cloud: AWS S3, Glue, Athena
Machine Learning: scikit-learn, MLflow
Backend: FastAPI
Frontend: Streamlit
LLM: Ollama (Llama 3.2)
Database: MySQL
📊 Dataset
Amazon Reviews 2023 (Cell Phones & Accessories)
~20.8 Million Reviews (~11GB)
~20.5 Million Cleaned Records
Fraud Rate: ~3.57%
🤖 Models & Performance
Sentiment Model
TF-IDF + Logistic Regression
Weighted F1: 0.841
Accuracy: 0.81
Fraud Detection Model
TF-IDF + Behavioral Features
ROC-AUC: 0.845
Best Threshold: 0.80
F1 Score: 0.356
Calibration
Brier Score: 0.143 → 0.029
ECE: 0.302 → 0.000146
⚡ Real-Time Streaming
⏱️ Latency: ~5 seconds
🚀 Throughput: ~150 msgs/sec
🔌 API Endpoints
Method	Endpoint	Description
GET	/healthz	Health check
GET	/metadata	Model info
POST	/predict	Score single review
POST	/predict/batch	Batch scoring
GET	/aggregates/products	Top products
GET	/aggregates/fraud-reviewers	Suspicious reviewers
GET	/stream/recent	Live stream results
🤖 LLM Features
Generates fraud explanations using Llama 3.2
Detects:
Duplicate text
Reviewer anomalies
Rating manipulation
Includes AI review auditor agent for product risk analysis
⏱️ Runtime
Stage	Time
Ingestion	~3.5 min
ETL	~24 min
Sentiment Training	~52 min
Fraud Training	~3.5 hrs
Total Pipeline	~6 hours

Runs on a 16GB laptop.

📂 Project Structure
FraudLens/
│── src/
│   ├── etl/
│   ├── models/
│   ├── stream/
│   ├── serve/
│   ├── agents/
│── data/
│── tests/
│── README.md
🚀 Getting Started
# Clone repository
git clone https://github.com/Paruchuri-Rajesh/Large_Scale_Product-Review.git

cd Large_Scale_Product-Review

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app:app --reload
⚠️ Limitations
Single product category dataset
No true fraud labels (weak supervision used)
Linear models only
Missing helpful vote data
🔮 Future Work
Multi-category expansion
Human-labeled fraud dataset
Docker deployment
Real-time feature store
Advanced ML models
