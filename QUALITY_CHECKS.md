# Data Quality & Validation Checks

## 1. Purpose

This document defines **data quality checks and validation rules** used throughout the pipeline to ensure:

- Reliability of downstream features and models
- Early detection of data pipeline issues
- Prevention of silent failures and data leakage

The goal is **risk reduction**, not exhaustive validation.

Checks are applied at each data layer (Bronze, Silver, Gold) with different severity levels.

---

## 2. Quality Philosophy

- Not all data issues should fail a pipeline
- Some issues should **warn**, others should **stop execution**
- Missingness is often **informative**, not an error
- Temporal correctness is more important than value perfection

Checks are designed to be **interpretable, lightweight, and actionable**.

---

## 3. Severity Levels

| Level | Description | Action |
|---|---|---|
| **ERROR** | Violates core assumptions | Pipeline fails |
| **WARN** | Unexpected but tolerable | Logged for review |
| **INFO** | Informational metrics | Logged for monitoring |

---

## 4. Bronze Layer Checks (Raw Data)

**Objective:** Ensure raw synthetic data is structurally valid and parsable.

### Schema & Identity
- **ERROR:** Required columns missing (patient_id, admission_id, event_time)
- **ERROR:** event_time cannot be parsed as timestamp
- **WARN:** Duplicate rows detected (allowed in Bronze, counted)

### Temporal Validity
- **ERROR:** event_time outside admission window
- **WARN:** event_time in future relative to pipeline run

### Value Ranges (Soft Checks)
- **WARN:** SpO₂ outside 0–100
- **WARN:** Temperature outside 30–45 °C
- **WARN:** HR outside 20–220 bpm
- **WARN:** SBP outside 50–250 mmHg
- **WARN:** Lactate outside 0–15 mmol/L

> Rationale: Bronze preserves raw structure; outliers are flagged, not removed.

---

## 5. Silver Layer Checks (Cleaned & Standardized)

**Objective:** Guarantee consistent, analysis-ready event data.

### Deduplication
- **ERROR:** Duplicate keys remain after dedupe logic  
  (patient_id, admission_id, event_time)
- **INFO:** Number of rows removed during deduplication

### Timestamp Consistency
- **ERROR:** Null event_time after standardization
- **ERROR:** admission_time ≥ discharge_time (when discharge exists)

### Unit Consistency
- **ERROR:** Mixed units detected within same signal
- **INFO:** Confirmed standard units per column

### Missingness Tracking
- **INFO:** % missing per vital/lab
- **WARN:** Sudden increase in missingness (>X% change vs baseline)

> Rationale: Silver guarantees *consistency*, not completeness.

---

## 6. Gold Layer Checks (Anchors, Features, Labels)

**Objective:** Protect modeling integrity and prevent leakage.

---

### 6.1 Anchor-Time Validation

- **ERROR:** Duplicate (patient_id, admission_id, anchor_time)
- **ERROR:** anchor_time outside admission window
- **ERROR:** anchor_time ≥ outcome_time (if outcome exists)
- **INFO:** Anchor count per admission (distribution)

---

### 6.2 Feature Construction Checks

#### Window Integrity
- **ERROR:** Feature window includes events after anchor_time
- **ERROR:** Window boundaries incorrectly defined (e.g., negative duration)

#### Feature Completeness
- **INFO:** % null per feature column
- **WARN:** Feature entirely null across dataset

#### Missingness Features
- **ERROR:** time_since_last_measurement < 0
- **WARN:** time_since_last_measurement exceeds window size unexpectedly

> Rationale: Missing features are allowed; broken windows are not.

---

### 6.3 Labeling & Leakage Prevention

- **ERROR:** Labels generated using future feature data
- **ERROR:** Outcome_time ≤ anchor_time included in label window
- **INFO:** Positive label rate (class imbalance)
- **WARN:** Label prevalence deviates significantly from expected synthetic design

> This is the **most critical validation area**.

---

## 7. Train/Test Split Validation

- **ERROR:** Same patient_id appears in both train and test sets
- **INFO:** Number of patients per split
- **INFO:** Outcome rate per split (sanity check)

---

## 8. Model Output Sanity Checks

- **ERROR:** risk_score outside [0,1]
- **WARN:** All predictions identical (collapsed model)
- **INFO:** Risk score distribution (min, max, mean)

---

## 9. Monitoring Metrics (Logged per Run)

Even though this is a batch project, the following metrics are logged for each run:

- Row counts in/out per layer
- % missing per key signal
- Anchor count per patient
- Positive label rate
- Model performance summary (AUROC, AUPRC)

These metrics enable **regression detection** across runs.

---

## 10. Known Limitations

- Synthetic data quality checks may not capture all real EHR failure modes
- No automated alerting or rollback mechanisms are implemented
- Thresholds are heuristic and illustrative

---

## 11. Summary

These quality checks are designed to:

- Catch structural and temporal errors early
- Preserve informative missingness
- Protect against data leakage
- Support confidence in downstream modeling outputs

The emphasis is on **engineering judgment and transparency**, not exhaustive enforcement.

