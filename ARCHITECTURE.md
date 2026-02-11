# Architecture — Clinical Deterioration Risk Monitoring (Batch Early Warning)

## 1. Objective

Build a batch analytics + ML pipeline that produces **risk scores** (0–1) for inpatients indicating likelihood of **clinical deterioration in the next 24–48 hours**, based only on data available up to the scoring time.

This system is designed for **monitoring and prioritization** (early warning) and does not provide diagnosis or treatment recommendations.

---

## 2. High-Level System Flow

**End-to-end pipeline (batch):**

1) Synthetic event generation  
2) Bronze ingestion (raw events)  
3) Silver transformation (clean + standardized events)  
4) Gold feature + label construction (time-windowed)  
5) Model training & evaluation  
6) Batch scoring (inference)  
7) Monitoring outputs for dashboarding

---

## 3. Architecture Diagram (Logical)

flowchart LR
  subgraph GEN[Data Generation]
    A1[patients_raw]:::data
    A2[vitals_events_raw]:::data
    A3[labs_events_raw]:::data
    A4[outcomes_raw]:::data
  end

  subgraph BRONZE[Bronze\n(raw + validated)]
    B1[patients_bronze]:::data
    B2[vitals_events_bronze]:::data
    B3[labs_events_bronze]:::data
    B4[outcomes_bronze]:::data
  end

  subgraph SILVER[Silver\n(clean + standardized)]
    C1[patients_silver]:::data
    C2[vitals_events_silver]:::data
    C3[labs_events_silver]:::data
    C4[outcomes_silver]:::data
  end

  subgraph GOLD[Gold\n(anchored + featured)]
    D1[anchors_gold\n(patient_id, anchor_time)]:::data
    D2[features_gold\n(window stats + trends + missingness)]:::data
    D3[labels_gold\n(24–48h deterioration)]:::data
    D4[train_dataset_gold\n(features + label)]:::data
    D5[score_dataset_gold\n(features only)]:::data
  end

  subgraph ML[Modeling]
    E1[Train LR baseline]:::proc
    E2[Train GBT]:::proc
    E3[Evaluate\nAUROC/AUPRC\nalert-rate/lead-time\ncalibration]:::proc
    E4[Model artifact + metrics]:::out
  end

  subgraph INF[Inference]
    F1[Batch scoring]:::proc
    F2[risk_scores\n(patient_id, anchor_time, risk_score, band)]:::out
  end

  subgraph BI[Consumption]
    G1[Dashboard dataset\n(top risk, trends, distribution)]:::out
  end

  A1 --> B1
  A2 --> B2
  A3 --> B3
  A4 --> B4

  B1 --> C1
  B2 --> C2
  B3 --> C3
  B4 --> C4

  C1 --> D1
  C2 --> D2
  C3 --> D2
  C4 --> D3

  D1 --> D4
  D2 --> D4
  D3 --> D4

  D5 --> F1
  F1 --> F2 --> G1

  D4 --> E1 --> E3
  D4 --> E2 --> E3 --> E4

  classDef data fill:#eef,stroke:#446,stroke-width:1px;
  classDef proc fill:#efe,stroke:#363,stroke-width:1px;
  classDef out fill:#fee,stroke:#844,stroke-width:1px;

---

## 4. Runtime & Execution Model

- **Execution type:** Batch (offline)
- **Processing engine:** PySpark (Spark SQL + Spark ML)
- **Storage format:** Parquet
- **Granularity:** Predictions are computed at **anchor times** (e.g., hourly) per patient during admission.
- **Partitioning:** By date and/or patient_id for scalable reads/writes.

---

## 5. Data Layers and Contracts

### 5.1 Bronze Layer (Raw)
**Purpose:** Preserve raw event structure with minimal transformations.

**Tables (Parquet):**
- `patients_bronze`
- `vitals_events_bronze`
- `labs_events_bronze`
- `outcomes_bronze`

**Contract expectations:**
- Raw timestamps may be messy
- Duplicate events may exist
- Missing values may exist
- Values may be out-of-range (to be flagged)

**Quality checks (examples):**
- required columns non-null (patient_id, event_time)
- timestamp parse success rate
- duplicate detection rates
- basic range checks (e.g., SpO₂ 0–100)

---

### 5.2 Silver Layer (Cleaned & Standardized)
**Purpose:** Convert raw events into clean, consistent, analysis-ready events **without collapsing time**.

**Key transformations:**
- Normalize timestamp types (UTC or consistent local time)
- Standardize units (if applicable)
- Dedupe using deterministic keys (patient_id, event_time, measurement_type)
- Keep missingness explicit (do not “hide” missing values)
- Align schema for vitals and labs events

**Tables:**
- `patients_silver`
- `vitals_events_silver`
- `labs_events_silver`
- `outcomes_silver`

**Contract expectations:**
- event_time is valid and consistent
- duplicates removed (or flagged)
- columns standardized across runs

---

### 5.3 Gold Layer (Features + Labels)
**Purpose:** Create **anchor-time snapshots** and compute features using **rolling windows** while enforcing leakage prevention.

**Gold datasets:**
- `anchors_gold`  
  One row per (patient_id, anchor_time)
- `features_gold`  
  One row per (patient_id, anchor_time) with engineered features
- `labels_gold`  
  One row per (patient_id, anchor_time) with deterioration label
- `train_dataset_gold` (features + label)
- `score_dataset_gold` (features only, for inference)

---

## 6. Anchor-Time Design (Core Concept)

### Definition
An **anchor time** is a fixed point during a patient’s admission when the model generates a risk score.

**Example approach:**
- Create anchor times at hourly cadence:
  - from admission_time + warmup_period
  - to discharge_time (or outcome_time - buffer)

**Design rationale:**
- Matches real monitoring workflows (periodic reassessment)
- Enables time-window feature engineering
- Supports lead-time evaluation

---

## 7. Feature Engineering (Time-Windowed)

For each anchor_time, compute features from preceding time windows:

- **W6:** (anchor_time - 6h, anchor_time]
- **W12:** (anchor_time - 12h, anchor_time]
- **W24:** (anchor_time - 24h, anchor_time]

### Feature families (examples)

**A) Summary statistics**
- last_value, mean, min, max, std for each vital/lab

**B) Trend features**
- delta = last - first (per window)
- slope via simple linear fit approximation (optional)
- rate-of-change indicators

**C) Volatility**
- std / range within window

**D) Missingness & measurement density**
- measurement_count per window
- time_since_last_measurement

**E) Simple derived features (optional)**
- shock_index = HR / SBP
- pulse_pressure = SBP - DBP

---

## 8. Labeling Logic (Leakage-Safe)

### Label definition
For each (patient_id, anchor_time):

`label = 1` if a deterioration outcome occurs in:
- **(anchor_time + 24h, anchor_time + 48h]**

otherwise `label = 0`

### Leakage prevention rules
- Features must only use events with `event_time <= anchor_time`
- Anchors must exclude post-outcome intervals:
  - Do not create anchors after outcome_time
- If outcome happens too soon after anchor (e.g., within 0–24h), label remains 0 under the 24–48h definition (or excluded if you choose a different framing—must be documented and consistent).

---

## 9. Model Training Architecture

### Models
1) **Logistic Regression (baseline)**
- interpretable, stable
- used to sanity-check feature behavior

2) **Gradient-Boosted Trees (GBT)**
- captures non-linearities
- typically higher performance on tabular time-windowed features

### Train/Test split
- Split by **patient_id** (no patient appears in both sets)
- Optional extra-depth: temporal split (earlier admissions train, later admissions test)

### Outputs
- trained model artifact
- feature importance / coefficients
- evaluation report artifacts

---

## 10. Evaluation Architecture (Operational Metrics)

Beyond AUROC, evaluate in ways aligned to alerting:

- **AUPRC** (critical for rare events)
- **Sensitivity at fixed alert rate**
  - Example: top 5% highest risk anchors → what % of deteriorations caught?
- **Lead-time**
  - time between first alert and outcome_time
- **Calibration**
  - reliability across risk bins

---

## 11. Batch Scoring (Inference)

### Input
- `score_dataset_gold` for a scoring period

### Output tables
- `risk_scores`
  - patient_id
  - anchor_time
  - risk_score (0–1)
  - risk_band (low/med/high)
  - top_features (optional, for explainability)

These outputs are intended to feed dashboards or downstream monitoring.

---

## 12. Non-Goals (Explicit)

This project intentionally does NOT include:
- streaming ingestion (Kafka)
- real-time alert delivery systems
- clinical decision support integration
- deployment to EHR systems
- use of real patient data

---

## 13. Implementation Notes

- Use Parquet for each layer to keep the pipeline inspectable.
- Keep notebooks optional; core logic must live in Python modules.
- Every transformation step should log:
  - record counts in/out
  - % missing for key columns
  - duplicate removal counts
  - range violation counts

---

## 14. Repo Mapping (Where Each Piece Lives)

- Synthetic data generator: `data_generation/`
- ETL jobs: `etl/`
- Feature engineering modules: `features/`
- Label generation: `labeling/`
- Training + evaluation: `modeling/`
- Batch scoring: `inference/`
- Docs: root markdown files

---
