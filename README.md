# FraudLens: Large-Scale Product Review Sentiment and Fraud Detection

A distributed batch-and-streaming pipeline for large-scale analysis of Amazon product reviews. FraudLens ingests over 20 million records, performs feature engineering with Apache Spark, trains sentiment and fraud detection models, and serves predictions through five heterogeneous serving surfaces — all running end-to-end in roughly six hours on a standard 16 GB laptop.

**Team (Group 6):** Prerana Ramesh, Rajesh Paruchuri, Ritika Mukesh Neema, Sneha Singh
**Department of Applied Data Science, San Jose State University**

**Source code:** https://github.com/Paruchuri-Rajesh/Large_Scale_Product-Review

## Overview

Online reviews shape what people buy on Amazon, and they also shape which sellers thrive. With hundreds of millions of reviews across thousands of categories, manual moderation is impossible. FraudLens tackles two related ML problems at scale:

1. **Sentiment classification** — determining whether a review expresses a positive, neutral, or negative opinion
2. **Fraud detection** — identifying synthetic, incentivized, or duplicated reviews that inflate or deflate product ratings

### Key Results

| Model | Metric | Score |
|-------|--------|-------|
| Sentiment | Weighted F1 | **0.841** |
| Sentiment | Accuracy | 0.810 |
| Fraud | ROC-AUC | **0.845** |
| Fraud (t=0.80) | F1 | 0.356 |
| Fraud (calibrated) | Brier Score | 0.029 (down from 0.143) |
| Fraud (calibrated) | ECE | 0.000146 (down from 0.302) |

### Main Contributions

- A five-stage distributed pipeline (ingest → ETL → train → stream → serve) on a single commodity laptop
- A reproducible leakage-prevention strategy for weak fraud labels via controlled label noise
- A threshold-tuning study showing F1-optimal operating points differ substantially from default
- Post-hoc isotonic calibration that reduces ECE by three orders of magnitude with no ranking degradation
- Five heterogeneous serving surfaces, including an MCP tool interface and an autonomous LLM auditor

## Architecture

The system runs as five sequential stages. Each stage writes its output to disk (Parquet, JSONL, or joblib) so any stage can be re-run independently.
