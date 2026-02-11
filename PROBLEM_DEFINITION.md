# Problem Definition — Clinical Deterioration Risk Monitoring

## 1. Background & Context

Hospitalized patients are routinely monitored using vital signs, laboratory tests, and clinical observations.  
Despite this monitoring, **some patients experience unexpected clinical deterioration** during their hospital stay, leading to:

- Unplanned ICU transfers  
- Rapid Response Team (RRT) activations  
- Increased operational burden and patient risk  

In many cases, deterioration is **preceded by subtle trends** in physiological measurements that may not be immediately obvious through manual review or threshold-based rules.

Healthcare operations teams therefore seek **early warning systems** that help identify patients who may require closer monitoring *before* escalation occurs.

---

## 2. Business Problem

Clinical teams must continuously decide **which patients require increased attention** under time and staffing constraints.

Current challenges include:
- Large patient volumes with limited staff attention
- Manual review of vitals and labs that is time-consuming
- Alert fatigue from simplistic threshold-based rules
- Limited ability to detect **trend-based deterioration**

The business need is **prioritization**, not diagnosis.

---

## 3. Problem Statement

> **Can we estimate the short-term risk of clinical deterioration for hospitalized patients using recent trends in vital signs and laboratory data, in order to support early warning and patient prioritization?**

This problem is framed as a **risk estimation task**, not a clinical decision-making system.

---

## 4. Target Outcome (Label Definition)

For the purposes of this project, *clinical deterioration* is defined as the occurrence of one of the following events:

- Unplanned ICU transfer  
- Rapid Response Team (RRT) activation  

An outcome is considered positive if it occurs **within 24–48 hours after a prediction time**.

This outcome definition is:
- Operationally meaningful
- Commonly used in early warning research
- Non-diagnostic in nature

---

## 5. Prediction Objective

At predefined points during a patient’s hospital admission, the system will estimate:

> **The probability that a patient will experience clinical deterioration within the next 24–48 hours.**

### Prediction Characteristics
- **Prediction type:** Binary classification  
- **Model output:** Continuous risk score (0–1)  
- **Prediction horizon:** 24–48 hours  
- **Prediction frequency:** Periodic (anchor-based during admission)

The output is a **risk score**, not a yes/no decision.

---

## 6. Intended Use

This system is intended to:
- Support early warning and monitoring workflows
- Help prioritize patients for closer observation
- Provide interpretable signals derived from recent physiological trends

This system is **not intended** to:
- Diagnose diseases or conditions  
- Recommend treatments or interventions  
- Replace clinical judgment  

All data used in this project is **synthetic** and generated solely for demonstration and learning purposes.

---

## 7. Data Scope

The system uses simulated hospital data designed to resemble real-world electronic health record (EHR) characteristics:

### Data Categories
- **Vitals:** Heart rate, blood pressure, respiratory rate, temperature, SpO₂  
- **Labs:** WBC, creatinine, lactate (synthetic but clinically plausible)  
- **Demographics:** Age, sex  
- **Comorbidities:** Binary indicators (e.g., diabetes, CKD)  
- **Temporal behavior:** Irregular measurement intervals and missing data  

The model relies not only on raw values, but also on **recent trends and variability**.

---

## 8. Modeling Assumptions

- Patient deterioration risk is reflected in **recent physiological trends**, not single measurements
- Missingness and measurement frequency may carry informative signal
- Only data available **up to the prediction time** may be used
- Each patient admission is treated as an independent observation unit

---

## 9. Evaluation Philosophy

Model performance is evaluated based on its usefulness for operational monitoring rather than purely statistical accuracy.

Key evaluation considerations include:
- Ability to rank patients by risk
- Sensitivity at fixed alert rates (e.g., top 5% highest risk)
- Lead time between alert and deterioration event
- Calibration of predicted risk scores

The goal is to enable **actionable early warning with manageable alert burden**.

---

## 10. Success Criteria

The project is considered successful if it demonstrates that:
- Recent patient trends can be transformed into meaningful risk signals
- A subset of high-risk patients can be identified ahead of deterioration
- The system design avoids data leakage and reflects real-world constraints
- Risk drivers are interpretable and clinically intuitive

---

## 11. Constraints & Limitations

- All data is synthetic and does not represent real patient outcomes
- Results are not clinically validated
- The system is batch-based and not real-time
- No integration with clinical systems or workflows is performed

---

## 12. Ethical & Safety Considerations

- No real patient data is used
- Outputs are clearly positioned as non-diagnostic
- Model limitations and assumptions are explicitly documented
- Interpretability and transparency are prioritized

---

## 13. Summary

This project focuses on building a **responsible, time-aware early warning system** that demonstrates how data engineering, feature design, and machine learning can be combined to support patient monitoring and prioritization in a healthcare setting—without making clinical decisions.

The emphasis is on **system design, rigor, and operational realism**, rather than purely predictive performance.
