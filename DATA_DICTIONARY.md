# Data Dictionary — Clinical Deterioration Risk Monitoring

## 1. Overview

This project uses **synthetic hospital-like inpatient data** organized as event tables (irregular time series) and derived analytical layers.

### Data Layering
- **Bronze:** raw synthetic event tables (minimal transforms, basic validation)
- **Silver:** cleaned + standardized events (schema stable, deduped, aligned timestamps)
- **Gold:** anchor-time snapshots, engineered features, leakage-safe labels, and model-ready datasets

### Conventions
- **Time zone:** All timestamps stored as UTC (recommended) or a single consistent timezone.
- **Primary keys:** Synthetic IDs; uniqueness rules documented per table.
- **Null handling:** Missingness is expected and preserved; missingness-related features are created in Gold.
- **Units:** Synthetic but clinically plausible; units are consistent within Silver/Gold.

---

## 2. Core Entities & Relationships

### Entity relationship summary
- One `patient_id` → many `vitals_events` rows
- One `patient_id` → many `labs_events` rows
- One `patient_id` → zero/one deterioration event in `outcomes` (for simplicity)
- One `patient_id` → many `anchor_time` rows in Gold
- One `(patient_id, anchor_time)` → one feature row and one label row

---

## 3. Bronze Tables (Raw)

### 3.1 `patients_bronze`
**Description:** One row per patient admission.

| Column | Type | Nullable | Example | Notes |
|---|---|---:|---|---|
| patient_id | string | No | "P000123" | Unique synthetic identifier |
| admission_id | string | No | "A000123" | One admission per patient in v1 (can expand later) |
| admission_time | timestamp | No | 2026-01-10 14:00:00 | Admission start |
| discharge_time | timestamp | Yes | 2026-01-14 11:00:00 | Discharge/end; may be null for ongoing stays |
| age | int | No | 67 | 0–100 (synthetic) |
| sex | string | Yes | "F" | Optional |
| diabetes_flag | int | Yes | 0/1 | Comorbidity flags |
| ckd_flag | int | Yes | 0/1 |  |
| copd_flag | int | Yes | 0/1 |  |
| chf_flag | int | Yes | 0/1 |  |
| created_at | timestamp | Yes | 2026-01-10 14:01:00 | Optional lineage |

**Constraints / checks**
- `admission_time < discharge_time` when discharge_time present
- age within plausible range
- patient_id + admission_id unique

---

### 3.2 `vitals_events_bronze`
**Description:** Irregular event stream of bedside vitals.

| Column | Type | Nullable | Example | Notes |
|---|---|---:|---|---|
| patient_id | string | No | "P000123" | FK to patients |
| admission_id | string | No | "A000123" | FK to patients |
| event_time | timestamp | No | 2026-01-11 09:15:00 | Time vitals recorded |
| hr | double | Yes | 102.0 | Heart rate (bpm) |
| sbp | double | Yes | 98.0 | Systolic BP (mmHg) |
| dbp | double | Yes | 62.0 | Diastolic BP (mmHg) |
| rr | double | Yes | 22.0 | Respiratory rate |
| temp_c | double | Yes | 38.1 | Temperature in Celsius |
| spo2 | double | Yes | 92.0 | Oxygen saturation (%) |
| source_system | string | Yes | "synthetic_monitor" | Optional |
| ingest_run_id | string | Yes | "run_2026_01_12" | Optional lineage |

**Constraints / checks**
- Required: patient_id, admission_id, event_time
- Ranges (soft checks): spo2 0–100, temp_c 30–45, hr 20–220, sbp 50–250, rr 5–60
- Duplicates possible in Bronze (handled in Silver)

---

### 3.3 `labs_events_bronze`
**Description:** Irregular lab results (less frequent than vitals).

| Column | Type | Nullable | Example | Notes |
|---|---|---:|---|---|
| patient_id | string | No | "P000123" | FK |
| admission_id | string | No | "A000123" | FK |
| event_time | timestamp | No | 2026-01-11 06:00:00 | Specimen/result time |
| wbc | double | Yes | 14.2 | White blood cell count |
| creatinine | double | Yes | 1.8 | Renal function |
| lactate | double | Yes | 2.6 | Tissue perfusion marker |
| sodium | double | Yes | 134.0 | Optional |
| potassium | double | Yes | 4.8 | Optional |
| bun | double | Yes | 28.0 | Optional |
| source_system | string | Yes | "synthetic_lab" | Optional |
| ingest_run_id | string | Yes | "run_2026_01_12" | Optional lineage |

**Constraints / checks**
- Required: patient_id, admission_id, event_time
- Soft ranges: wbc 0–50, creatinine 0.2–10, lactate 0–15

---

### 3.4 `outcomes_bronze`
**Description:** Deterioration events during admission.

| Column | Type | Nullable | Example | Notes |
|---|---|---:|---|---|
| patient_id | string | No | "P000123" | FK |
| admission_id | string | No | "A000123" | FK |
| outcome_time | timestamp | No | 2026-01-12 18:30:00 | Time of event |
| outcome_type | string | No | "ICU_TRANSFER" | Enum: ICU_TRANSFER, RRT |
| outcome_flag | int | No | 1 | Always 1 per row (for clarity) |

**Constraints / checks**
- outcome_time must fall within admission window
- In v1, at most one outcome per admission (can expand later)

---

## 4. Silver Tables (Cleaned & Standardized)

> Silver tables mirror Bronze schemas but enforce stable typing, timestamp consistency, deduplication rules, and unit consistency.

### 4.1 `patients_silver`
Same columns as Bronze, with:
- standardized `sex` values (e.g., "M","F","U")
- validated admission/discharge ordering
- one row per (patient_id, admission_id)

### 4.2 `vitals_events_silver`
Same columns as Bronze, with:
- `event_time` normalized to consistent timezone
- duplicates removed using a deterministic rule:
  - key: (patient_id, admission_id, event_time)
  - keep latest record (if synthetic generator creates duplicates)
- unit consistency guaranteed (temp always Celsius)

### 4.3 `labs_events_silver`
Same columns as Bronze, with:
- dedupe rule on (patient_id, admission_id, event_time)
- stable numeric types (double)
- unit consistency guaranteed

### 4.4 `outcomes_silver`
Same columns as Bronze, with:
- validated outcome_type enumeration
- validated outcome_time within admission window

---

## 5. Gold Tables (Anchors, Features, Labels)

### 5.1 `anchors_gold`
**Description:** Anchor-time grid for periodic scoring.

| Column | Type | Nullable | Example | Notes |
|---|---|---:|---|---|
| patient_id | string | No | "P000123" | FK |
| admission_id | string | No | "A000123" | FK |
| anchor_time | timestamp | No | 2026-01-12 10:00:00 | Scoring time |
| anchor_seq | int | Yes | 12 | Optional index |
| warmup_hours | int | Yes | 6 | Optional metadata |
| is_valid_anchor | int | Yes | 1 | 1 if within allowed window |

**Constraints**
- Unique: (patient_id, admission_id, anchor_time)
- Anchors must be within admission window and before outcome_time (if outcome exists)

---

### 5.2 `features_gold`
**Description:** One row per (patient_id, admission_id, anchor_time) containing window-show features.

**Primary key:** (patient_id, admission_id, anchor_time)

#### Feature naming convention
`<signal>__<window>__<stat>`

- signal: hr, sbp, dbp, rr, temp_c, spo2, wbc, creatinine, lactate, etc.
- window: w6, w12, w24
- stat: last, mean, min, max, std, delta, slope, count, tslm (time_since_last_measurement)

#### Required identifier columns
| Column | Type | Nullable | Notes |
|---|---|---:|---|
| patient_id | string | No |  |
| admission_id | string | No |  |
| anchor_time | timestamp | No |  |

#### Example feature columns (starter set)
| Column | Type | Nullable | Notes |
|---|---|---:|---|
| hr__w6__last | double | Yes | Last HR value in (anchor-6h, anchor] |
| hr__w6__mean | double | Yes | Mean HR in window |
| hr__w6__std | double | Yes | HR volatility |
| hr__w6__delta | double | Yes | last - first |
| hr__w6__count | int | Yes | measurements count |
| hr__w6__tslm | double | Yes | hours since last measurement |
| spo2__w12__min | double | Yes | Min SpO₂ in 12h |
| sbp__w12__min | double | Yes | Min SBP in 12h |
| lactate__w24__max | double | Yes | Max lactate in 24h |
| wbc__w24__mean | double | Yes | Avg WBC in 24h |
| creatinine__w24__delta | double | Yes | Change in creatinine |
| shock_index__w6__last | double | Yes | HR/SBP using last values (if both present) |

**Notes**
- Missing features are allowed when no measurements exist in the window; missingness is informative.
- `__tslm` should cap at window size (e.g., max 6h for w6), or use null if no prior measure.

---

### 5.3 `labels_gold`
**Description:** Leakage-safe labels aligned to anchors.

| Column | Type | Nullable | Example | Notes |
|---|---|---:|---|---|
| patient_id | string | No | "P000123" |  |
| admission_id | string | No | "A000123" |  |
| anchor_time | timestamp | No | 2026-01-12 10:00:00 |  |
| label_deterioration_24_48h | int | No | 0/1 | 1 if outcome in (anchor+24h, anchor+48h] |
| outcome_time | timestamp | Yes | 2026-01-13 09:00:00 | Stored for evaluation only |
| outcome_type | string | Yes | ICU_TRANSFER | Stored for analysis |

**Constraints**
- Unique: (patient_id, admission_id, anchor_time)
- Label must be generated using only outcome metadata, not future features.

---

### 5.4 `train_dataset_gold`
**Description:** Final training dataset (features + label).

| Column group | Notes |
|---|---|
| identifiers | patient_id, admission_id, anchor_time |
| features | all feature columns from `features_gold` |
| target | label_deterioration_24_48h |

---

### 5.5 `score_dataset_gold`
**Description:** Dataset used for batch scoring (no label).

| Column group | Notes |
|---|---|
| identifiers | patient_id, admission_id, anchor_time |
| features | all feature columns from `features_gold` |

---

## 6. Scoring Output Tables

### 6.1 `risk_scores`
**Description:** Inference output for monitoring and dashboarding.

| Column | Type | Nullable | Example | Notes |
|---|---|---:|---|---|
| patient_id | string | No | "P000123" |  |
| admission_id | string | No | "A000123" |  |
| anchor_time | timestamp | No | 2026-01-12 10:00:00 |  |
| risk_score | double | No | 0.73 | 0–1 probability-like score |
| risk_band | string | Yes | "HIGH" | Derived from thresholds |
| model_version | string | Yes | "gbt_v1" | For reproducibility |
| scored_at | timestamp | Yes | 2026-02-10 09:00:00 | Scoring time |

Optional (extra depth):
- `top_drivers` (string/json) — top contributing features (LR coeffs or GBT importance proxy)

---

## 7. Enumerations

### outcome_type
- ICU_TRANSFER
- RRT

### sex (optional)
- M
- F
- U

### risk_band (optional)
- LOW
- MEDIUM
- HIGH

---

## 8. Data Quality Checks (Recommended)

Applied at each stage:

### Bronze checks
- schema present
- timestamp parse success
- duplicates rate
- range violations flagged

### Silver checks
- stable schema types
- dedupe complete
- admission/outcome time validity
- missingness % tracked

### Gold checks
- anchor grid validity (% valid anchors)
- feature null rates by window
- label prevalence (class imbalance)
- leakage audit:
  - max event_time used for features <= anchor_time

---
