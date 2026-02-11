# Model Card — Clinical Deterioration Risk Model

## 1. Model Overview

**Model name:** Clinical Deterioration Risk Model  
**Model type:** Binary classification (risk estimation)  
**Output:** Continuous risk score (0–1)  
**Prediction horizon:** 24–48 hours  
**Execution mode:** Batch (offline scoring)  

This model estimates the **short-term risk of clinical deterioration** for hospitalized patients based on recent trends in vitals and laboratory data.

The model is designed to support **early warning and patient prioritization** and does not provide diagnoses or treatment recommendations.

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

Only data available **up to the prediction time (anchor_time)** is used.

---

## 4. Model Outputs

- **risk_score:** A continuous value between 0 and 1 representing estimated risk
- **risk_band:** Optional categorical grouping (e.g., LOW / MEDIUM / HIGH)

The output represents **relative risk**, not a binary decision.

---

## 5. Data Used

### Data source
- Fully **synthetic** inpatient data generated for demonstration purposes

### Data characteristics
- Irregularly sampled vitals and labs
- Explicit missingness
- Simulated deterioration events (ICU transfer, RRT activation)

No real patient data is used in this project.

---

## 6. Modeling Approach

The project evaluates multiple models, including:
- Logistic Regression (interpretable baseline)
- Gradient-Boosted Trees (performance-oriented model)

Model selection prioritizes:
- Stable performance on imbalanced data
- Interpretability of risk drivers
- Alignment with operational evaluation metrics

> **Note:** Final model choice and hyperparameters will be documented after training.

---

## 7. Evaluation Strategy

Model performance is evaluated using metrics aligned with **early warning use cases**, including:
- AUROC
- AUPRC
- Sensitivity at fixed alert rates
- Lead time between alert and deterioration event
- Calibration of predicted risk scores

> **To be completed after model training.**

---

## 8. Performance Summary

> **To be completed after model training and evaluation.**

This section will include:
- Final evaluation metrics
- Selected alert thresholds
- Calibration results

---

## 9. Explainability & Interpretability

The project emphasizes interpretability through:
- Coefficient inspection for logistic regression
- Feature importance analysis for tree-based models
- Examination of key trend-based features

> **To be completed after model training.**

---

## 10. Ethical Considerations & Safety

- The model is not diagnostic and does not recommend treatment
- All data is synthetic
- Outputs are intended for monitoring and prioritization only
- Model limitations and uncertainty are explicitly documented

---

## 11. Limitations

- Synthetic data may not reflect full clinical complexity
- Model performance does not imply real-world effectiveness
- No prospective or clinical validation is performed
- Batch scoring does not support real-time alerting

---

## 12. Monitoring & Maintenance (Conceptual)

In a real deployment, the following would be monitored:
- Input data distributions
- Missingness patterns
- Prediction score distributions
- Model performance drift over time

> This project does not implement automated monitoring.

---

## 13. Versioning

- **Model version:** TBD
- **Training date:** TBD
- **Data version:** Synthetic v1

---

## 14. Additional Notes

This model card is intended to provide transparency into model purpose, scope, and limitations.  
It will be updated as modeling and evaluation are completed.

