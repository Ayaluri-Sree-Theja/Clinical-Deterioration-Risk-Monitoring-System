# Clinical Deterioration Risk Monitoring System

An end-to-end **data engineering and machine learning pipeline** for predicting short-term clinical deterioration risk and delivering operational insights through a monitoring dashboard.

This project simulates how modern healthcare analytics systems combine **data pipelines, machine learning models, and monitoring dashboards** to support early warning systems in hospital environments.

The system processes synthetic hospital event data, generates time-windowed clinical features, trains a predictive model, performs batch risk scoring, and exports results to a **Power BI command-center dashboard** used for monitoring and prioritization.

---

# Project Objectives

The goal of this project is to simulate a realistic **clinical deterioration early-warning system** from a data platform perspective.

The system demonstrates how to:

- Process irregular clinical event data
- Build time-windowed physiological features
- Train machine learning models for deterioration risk prediction
- Generate risk scores during a patient stay
- Deliver operational insights through a monitoring dashboard

The project is designed for **educational and portfolio demonstration purposes** using fully synthetic hospital data.

---

# Key Capabilities

This system implements several components typically found in production healthcare analytics platforms:

- End-to-end data engineering pipeline
- Time-window feature engineering
- Machine learning training and evaluation
- Batch inference pipeline
- Monitoring dashboard integration

Together these components simulate a **realistic operational risk monitoring system**.
# System Architecture

The project follows a **Medallion Architecture** commonly used in modern data platforms.
Synthetic Data
│
▼
RAW Layer
│
▼
BRONZE Layer (Ingestion)
│
▼
SILVER Layer (Cleaning & Standardization)
│
▼
GOLD Layer
├── Feature Engineering
├── Label Generation
└── Training Dataset
│
▼
Model Training & Evaluation
│
▼
Batch Risk Scoring
│
▼
Power BI Monitoring Dashboard

This architecture separates the system into clear processing stages:

| Layer | Purpose |
|------|--------|
| RAW | Synthetic source data |
| BRONZE | Raw ingestion with minimal transformation |
| SILVER | Cleaned and standardized datasets |
| GOLD | Feature engineering and modeling datasets |
| MODEL | Training and evaluation |
| SCORING | Risk score generation |
| ANALYTICS | Monitoring dashboard consumption |

The layered design improves:

- reproducibility
- debugging
- data lineage
- separation of responsibilities
# Key System Design Decisions

Several architectural decisions were made to simulate patterns used in real healthcare analytics systems.

## Medallion Data Architecture

The pipeline separates processing into **RAW → BRONZE → SILVER → GOLD layers**.

Benefits include:

- improved data lineage
- easier debugging
- reproducible feature pipelines
- separation between raw ingestion and analytics datasets

---

## Anchor-Based Prediction Framework

Instead of predicting risk once per hospital stay, the system generates **prediction anchors across time**.

This allows risk to be recalculated as new observations become available.

Benefits:

- supports continuous patient monitoring
- aligns with hospital early-warning workflows
- enables temporal feature engineering

---

## Time-Window Feature Engineering

Clinical signals are summarized over rolling windows:
# Key System Design Decisions

Several architectural decisions were made to simulate patterns used in real healthcare analytics systems.

## Medallion Data Architecture

The pipeline separates processing into **RAW → BRONZE → SILVER → GOLD layers**.

Benefits include:

- improved data lineage
- easier debugging
- reproducible feature pipelines
- separation between raw ingestion and analytics datasets

---

## Anchor-Based Prediction Framework

Instead of predicting risk once per hospital stay, the system generates **prediction anchors across time**.

This allows risk to be recalculated as new observations become available.

Benefits:

- supports continuous patient monitoring
- aligns with hospital early-warning workflows
- enables temporal feature engineering

---

## Time-Window Feature Engineering

Clinical signals are summarized over rolling windows:
6 hours
12 hours
24 hours

These windows capture both **recent physiological changes and longer-term trends**.

---

## Alert Rate–Based Monitoring

Rather than using a fixed probability threshold, the system evaluates model performance at **fixed alert rates**.

Example operational policies:

| Alert Rate | Operational Meaning |
|---|---|
| Top 5% | Highest risk patients flagged for immediate review |
| Top 10% | Moderate monitoring coverage |
| Top 20% | Expanded monitoring coverage |

This aligns model output with **clinical capacity constraints**.

### Directory Description

| Directory | Description |
|---|---|
| **checks** | Data quality validation scripts for bronze, silver, and gold layers |
| **configs** | Configuration files controlling pipeline parameters and feature generation |
| **dashboards** | Power BI command-center dashboard |
| **data** | Medallion architecture datasets (raw → bronze → silver → gold) |
| **jobs** | PySpark pipeline jobs executed sequentially |
| **notebooks** | Exploratory notebooks used during development |
| **outputs** | Generated prediction outputs and Power BI datasets |
| **reports** | Model artifacts and evaluation metrics |
| **scripts** | Local environment setup scripts |
| **docs (.md files)** | Project documentation including architecture and model card |

---
# Data Pipeline

The pipeline is implemented using **PySpark** and structured as modular jobs.

---

## 1. Synthetic Data Generation
jobs/00_synthesize_data.py

Generates synthetic hospital datasets including:

- patients
- admissions
- vitals events
- laboratory measurements
- deterioration outcomes

---

## 2. Bronze Layer — Raw Ingestion
jobs/10_bronze_ingest.py
Stores raw datasets with minimal transformation.

---

## 3. Silver Layer — Data Cleaning


jobs/20_bronze_to_silver.py


Applies data quality transformations including:

- schema normalization
- outlier clipping
- duplicate removal
- standardized event structure

---

## 4. Gold Layer — Feature Engineering


jobs/30_silver_to_gold_features.py


Creates model-ready features using time-window aggregations.

Example features include:

- heart rate trend indicators
- SpO₂ minimum values
- laboratory statistics
- measurement density indicators

---

## 5. Label Generation


jobs/40_build_labels.py


Defines deterioration labels based on simulated outcome events.

Prediction target:


Clinical deterioration within 24–48 hours


---

## 6. Training Dataset Creation


jobs/45_build_training_set.py


Combines features, labels, and admission attributes.

Dataset split:


Train: 70%
Validation: 15%
Test: 15%


Patient-level splitting is used to reduce leakage.
# Machine Learning Model

Two models were trained using **Spark ML**:

- Logistic Regression
- Gradient Boosted Trees

Logistic Regression was selected as the final model due to:

- higher AUPRC performance
- stable behavior under class imbalance
- improved interpretability

### Final Model Performance

| Metric | Value |
|------|------|
| AUROC | 0.701 |
| AUPRC | 0.060 |
| Baseline prevalence | ~2.3% |

Operational alert performance:

| Alert Rate | Precision | Recall | Lift |
|---|---|---|---|
| Top 5% | 9.22% | 20.0% | 4.00× |
| Top 10% | 7.37% | 32.0% | 3.20× |
| Top 20% | 5.26% | 45.7% | 2.28× |

Full model documentation is available in:


docs/MODEL_CARD.md


---

# Power BI Monitoring Dashboard

The pipeline exports a dataset for dashboard monitoring.


jobs/75_export_powerbi_dataset.py


Dataset produced:


pbi_risk_dataset.csv


The dashboard provides:

- hospital risk overview
- alert volume monitoring
- patient prioritization view
- operational system status indicators

Dashboard file:


dashboards/Clinical_Deterioration_Command_Center.pbix


---

# Running the Project

### Install dependencies


conda create -n clinical_ew python=3.10
conda activate clinical_ew
pip install pyspark pandas numpy pyyaml


---

### Run pipeline


python jobs/00_synthesize_data.py
python jobs/10_bronze_ingest.py
python jobs/20_bronze_to_silver.py
python jobs/30_silver_to_gold_features.py
python jobs/40_build_labels.py
python jobs/45_build_training_set.py
python jobs/50_train_sparkml.py
python jobs/60_evaluate.py
python jobs/70_batch_score.py
python jobs/75_export_powerbi_dataset.py


---

# Documentation

Detailed documentation is available in `/docs`.

| Document | Description |
|---|---|
| PROBLEM_DEFINITION.md | Project scope |
| ARCHITECTURE.md | System architecture |
| DATA_DICTIONARY.md | Feature definitions |
| QUALITY_CHECKS.md | Data validation rules |
| MODEL_CARD.md | Model documentation |

---

# Limitations

- Uses synthetic data only
- Not clinically validated
- Batch scoring only
- Demonstration project

---

# Author
Sree Theja Ayaluri

Project developed as part of a data engineering and analytics portfolio demonstrating:

- data platform design
- machine learning pipelines
- operational monitoring dashboards