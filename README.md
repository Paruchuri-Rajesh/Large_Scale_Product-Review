FraudLens: Large-Scale Review Sentiment and Fraud Detection
Overview

FraudLens is an end-to-end data engineering and machine learning system for analyzing Amazon product reviews at scale. It performs both sentiment classification and fraud detection while supporting real-time scoring through a streaming pipeline.

The system is designed to handle tens of millions of reviews using distributed processing and provides predictions through APIs, dashboards, and programmatic interfaces.

Problem Statement

Online product reviews significantly influence purchasing decisions, but they are often affected by spam, duplicate content, and incentivized reviews. Manual moderation is not scalable at this volume.

FraudLens addresses two key problems:

Sentiment classification (positive, neutral, negative)
Fraud detection (identifying suspicious or manipulated reviews)
System Architecture

The pipeline is organized into five stages:

Ingestion
Raw JSON review data stored in AWS S3
Converted to Parquet using AWS Glue
Batch Processing (Apache Spark)
Data cleaning and validation
Feature engineering (text, reviewer behavior, product aggregates)
Weak fraud label generation
Train/test split
Model Training
Sentiment model: TF-IDF + Logistic Regression
Fraud model: TF-IDF + behavioral features + Logistic Regression
Experiment tracking using MLflow
Leakage prevention using feature filtering and label noise
Streaming Pipeline
Kafka used for real-time ingestion
Spark Structured Streaming for micro-batch scoring
Predictions generated every few seconds
Serving Layer
FastAPI REST service
Dashboard for visualization
MCP server for tool-based access
LLM-based explanation system
Dataset
Source: Amazon Reviews 2023 (Cell Phones & Accessories)
Size: ~20.8 million reviews (~11 GB)
Cleaned: ~20.5 million records
Fraud class distribution: ~3.57%
Feature Engineering

Features are generated at multiple levels:

Text features
Review length, word count, punctuation usage
Reviewer-level features
Review frequency, rating patterns, verified purchase ratio
Product-level features
Review count, average rating, duplicate review detection

Weak fraud labels are created using heuristic rules. To avoid leakage:

Rule-based features are excluded from training
Controlled noise is introduced into labels
Model Performance
Sentiment Model
Algorithm: Logistic Regression with TF-IDF
Weighted F1 Score: 0.841
Accuracy: 0.81
Fraud Detection Model
ROC-AUC: 0.845
Best threshold: 0.80
F1 Score: 0.356
Calibration
Brier Score: reduced from 0.143 to 0.029
Expected Calibration Error: reduced from 0.302 to 0.000146
Real-Time Processing
Streaming framework: Kafka + Spark Structured Streaming
Processing mode: micro-batches (~5 seconds)
Throughput: ~150 messages/second
Supports both synchronous API scoring and asynchronous streaming
API Endpoints
Method	Endpoint	Description
GET	/healthz	Service health check
GET	/metadata	Model metadata and metrics
POST	/predict	Score a single review
POST	/predict/batch	Score multiple reviews
GET	/aggregates/products	Product-level aggregates
GET	/aggregates/fraud-reviewers	Suspicious reviewers
GET	/stream/recent	Latest streaming results
Technology Stack
Apache Spark (batch + streaming)
Apache Kafka (messaging)
AWS S3, Glue, Athena (data storage and ETL)
scikit-learn (modeling)
MLflow (experiment tracking)
FastAPI (serving)
Streamlit (dashboard)
MySQL (metadata storage)
Ollama / Llama 3.2 (LLM explanations)
Runtime
Stage	Duration
Ingestion	~3.5 minutes
ETL	~24 minutes
Sentiment training	~52 minutes
Fraud training	~3.5 hours
Full pipeline	~6 hours

Runs on a standard 16 GB machine.

Limitations
Dataset limited to a single product category
No ground-truth fraud labels (weak supervision used)
Linear models may not capture complex patterns
Missing helpful/unhelpful vote features
Future Work
Extend to multiple product categories
Incorporate human-labeled fraud data
Introduce deep learning models
Deploy using containerized infrastructure
Add real-time feature store
Repository

https://github.com/Paruchuri-Rajesh/Large_Scale_Product-Review
