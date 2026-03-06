# Model Card — Clinical Deterioration Risk Model

## 1. Model Overview

**Model name:** Clinical Deterioration Risk Model  
**Model type:** Binary classification (risk estimation)  
**Output:** Continuous risk score (0–1)  
**Prediction horizon:** 24–48 hours  
**Execution mode:** Batch (offline scoring)

This model estimates the **short-term risk of clinical deterioration** for hospitalized patients based on recent trends in vitals and laboratory data.

The model is designed to support **early warning and patient prioritization** and does not provide diagnoses or treatment recommendations.

The model operates within a **batch data pipeline that generates risk scores for downstream analytics and monitoring dashboards.**

---

## 2. Intended Use

### Intended users

- Clinical operations teams  
- Analytics teams supporting patient monitoring workflows  

### Intended use cases

- Identifying patients who may require closer observation  
- Prioritizing review by care teams  
- Supporting early warning monitoring dashboards  

### Out-of-scope uses

- Medical diagnosis  
- Treatment recommendations  
- Automated clinical decision-making  
- Use on real patient data without clinical validation  

---

## 3. Model Inputs

The model consumes **time-windowed features** derived from synthetic hospital data.

### Input categories

- Vital signs (e.g., heart rate, blood pressure, SpO₂)
- Laboratory values (e.g., WBC, creatinine, lactate)
- Demographics (e.g., age)
- Comorbidity indicators
- Trend and variability features over recent time windows (6h / 12h / 24h)
- Measurement density and missingness indicators

Only data available **up to the prediction time (`anchor_time`)** is used.

---

## 4. Model Outputs

- **risk_score:** Continuous value between 0 and 1 representing estimated deterioration risk
- **risk_band:** Optional categorical grouping (LOW / MEDIUM / HIGH)

The output represents **relative risk rather than a binary decision**.

---

## 5. Data Used

### Data source

Fully **synthetic inpatient data** generated for demonstration purposes.

### Data characteristics

- Irregularly sampled vitals and laboratory values
- Explicit missingness
- Simulated deterioration events (e.g., ICU transfer, rapid response activation)

No real patient data is used in this project.

### Dataset scale

The synthetic dataset used for this project includes approximately:

- ~2,200 unique patients
- ~3,000 simulated hospital admissions
- Thousands of prediction anchors across time windows
- Multiple vitals and laboratory measurements per admission

The dataset was designed to mimic **real-world clinical data characteristics**, including irregular measurement intervals and missing observations.

---

## 6. Modeling Approach

The project evaluates multiple models, including:

- Logistic Regression (interpretable baseline)
- Gradient-Boosted Trees (performance-oriented model)

Model selection prioritizes:

- Stable performance on imbalanced data
- Interpretability of risk drivers
- Alignment with operational evaluation metrics

### Final Model Selection

Two models were trained and evaluated using Spark ML:

- Logistic Regression (class-weighted)
- Gradient-Boosted Trees (GBT)

Logistic Regression was selected as the primary model due to:

- Higher AUPRC on the held-out test set
- Stable behavior under severe class imbalance (~2.3% prevalence)
- Better calibration characteristics
- Simpler interpretability for operational use

### Final Model Configuration (Logistic Regression)

- `maxIter`: 50
- `regParam`: 0.01
- `elasticNetParam`: 0.0
- Class weighting applied to address imbalance
- Feature vector assembled from windowed clinical features (6h / 12h / 24h)

Gradient-Boosted Trees were retained as a comparison model but not selected as the final model.

---

## 7. Evaluation Strategy

Evaluation is aligned with **early-warning system design** rather than traditional classification thresholds.

### Dataset Splits

- Train: 70%
- Validation: 15%
- Test: 15%
- Patient-level splitting used to reduce leakage across admissions.

### Primary Metrics

- AUROC
- AUPRC (primary model selection metric due to class imbalance)

### Operational Metrics

- Precision and Recall at fixed alert rates (Top 5%, 10%, 20%)
- Lift relative to baseline deterioration prevalence
- Calibration across risk bins

This evaluation emphasizes **operational usefulness rather than raw classification accuracy**.

---

## 8. Performance Summary

### Final Model: Logistic Regression

Test set performance:

- **AUROC:** 0.701
- **AUPRC:** 0.060
- **Baseline deterioration prevalence:** ~2.3%

Because deterioration events are rare, **precision values should be interpreted relative to baseline prevalence rather than absolute classification accuracy.**

### Operational Alert Performance (Test Set)

| Alert Rate | Precision | Recall | Lift |
|---|---|---|---|
| Top 5% | 9.22% | 20.0% | 4.00× |
| Top 10% | 7.37% | 32.0% | 3.20× |
| Top 20% | 5.26% | 45.7% | 2.28× |

**Interpretation**

At a 10% alert rate, the model captures approximately one-third of deterioration events while enriching event rate more than threefold relative to baseline prevalence.

These results are consistent with realistic early-warning system performance on imbalanced datasets.

---

## 9. Operational Decision Policy

The model produces a **continuous risk score between 0 and 1**.  
Operational alerting decisions are derived from this score using risk bands and alert-rate thresholds.

### Risk Bands

Risk scores are grouped into categorical bands to support monitoring dashboards and patient prioritization workflows.

| Risk Band | Description |
|----------|-------------|
| LOW | Patient shows minimal deterioration signals |
| MEDIUM | Patient exhibits moderate risk signals requiring observation |
| HIGH | Patient likely requires clinical review |

Risk band boundaries are configurable depending on operational monitoring capacity.

### Alert Rate Policy

Rather than using a fixed probability threshold, the system supports **alert rate–based prioritization**.

Example operational settings:

| Alert Rate | Operational Meaning |
|---|---|
| Top 5% | Highest-risk patients flagged for immediate review |
| Top 10% | Moderate monitoring load for care teams |
| Top 20% | Expanded monitoring coverage |

This approach allows the monitoring system to **adapt to available clinical capacity** while maintaining meaningful risk prioritization.

### Dashboard Integration

Risk scores and alert bands are exported to a monitoring dataset used by the **Power BI command-center dashboard**, where operational teams can:

- monitor hospital-wide deterioration risk
- track alert volume trends
- review high-risk patients requiring attention

---

## 10. Explainability & Interpretability

Explainability was emphasized through inspection of logistic regression coefficients and evaluation of feature behavior across time windows.

Key influential feature groups include:

- Recent heart rate trends
- Minimum SpO₂ values
- Lactate levels and laboratory trends
- Comorbidity indicators
- Measurement density features

The selected logistic regression model provides **transparent feature influence**, which supports interpretability compared to more complex tree-based models.

---

## 11. Ethical Considerations & Safety

- The model is **not diagnostic** and does not recommend treatment
- All data used in this project is synthetic
- Outputs are intended for **monitoring and prioritization only**
- Model limitations and uncertainty are explicitly documented

---

## 12. Limitations

- Synthetic data may not reflect full clinical complexity
- Model performance does not imply real-world effectiveness
- No prospective or clinical validation is performed
- Batch scoring does not support real-time alerting

---

## 13. Monitoring & Maintenance (Conceptual)

In a real deployment, the following would be monitored:

- Input data distributions
- Missingness patterns
- Prediction score distributions
- Model performance drift over time

For this project, **model outputs are monitored through a Power BI command-center dashboard built on the scored risk dataset.**

---

## 14. Versioning

- **Model version:** v1
- **Training configuration:** Logistic Regression (class-weighted)
- **Data version:** Synthetic v1
- **Pipeline:** Spark batch pipeline (RAW → GOLD → Batch Scoring)

---

## 15. Additional Notes

This model card documents the **final configuration and evaluation of the model used in this project.**

The document is intended to provide transparency into model purpose, scope, performance characteristics, and operational limitations.