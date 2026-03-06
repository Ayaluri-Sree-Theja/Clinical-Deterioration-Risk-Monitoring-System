# jobs/00_synthesize_data.py
import os
import math
import yaml
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, DateType
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_spark(project_cfg: dict) -> SparkSession:
    spark_cfg = project_cfg["spark"]
    checkpoint_dir = project_cfg["paths"]["checkpoints"]

    spark = (
        SparkSession.builder
        .master(spark_cfg.get("master", "local[2]"))
        .appName(project_cfg["project"]["name"] + "_synth")
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "2g"))
        .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 16)))
        .config("spark.default.parallelism", str(spark_cfg.get("default_parallelism", 16)))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    return spark


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_patient_ids(n: int) -> np.ndarray:
    # deterministic friendly ids
    return np.array([f"P{str(i).zfill(6)}" for i in range(1, n + 1)], dtype=object)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def choose_comorbidity(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """
    Simple comorbidity flags; keep it light but realistic.
    """
    # prevalences (rough, not clinical truth)
    diabetes = rng.binomial(1, 0.18, size=n)
    ckd = rng.binomial(1, 0.10, size=n)
    copd = rng.binomial(1, 0.12, size=n)
    chf = rng.binomial(1, 0.08, size=n)
    return pd.DataFrame({
        "diabetes": diabetes.astype(int),
        "ckd": ckd.astype(int),
        "copd": copd.astype(int),
        "chf": chf.astype(int),
        "comorbidity_count": (diabetes + ckd + copd + chf).astype(int)
    })


def synthesize_admissions(rng: np.random.Generator, synth_cfg: dict) -> pd.DataFrame:
    patients_n = int(synth_cfg["scale"]["patients"])
    admissions_n = int(synth_cfg["scale"]["admissions"])
    los_min = int(synth_cfg["scale"]["los_days_min"])
    los_max = int(synth_cfg["scale"]["los_days_max"])

    patient_ids = make_patient_ids(patients_n)

    # Age: mixture with tail
    ages = np.clip(rng.normal(58, 18, size=patients_n), 18, 95).round().astype(int)
    sex = rng.choice(["F", "M"], size=patients_n, p=[0.52, 0.48])

    comorb = choose_comorbidity(rng, patients_n)
    patients = pd.DataFrame({
        "patient_id": patient_ids,
        "age": ages,
        "sex": sex
    }).join(comorb)

    # Admissions sampled from patients (some patients have multiple)
    adm_patient = rng.choice(patient_ids, size=admissions_n, replace=True)
    admission_ids = np.array([f"A{str(i).zfill(7)}" for i in range(1, admissions_n + 1)], dtype=object)

    # Start times over a synthetic month
    base = datetime(2025, 1, 1, 0, 0, 0)
    start_offsets_hours = rng.integers(0, 24 * 30, size=admissions_n)
    admit_times = np.array([base + timedelta(hours=int(h)) for h in start_offsets_hours])

    los_days = rng.integers(los_min, los_max + 1, size=admissions_n)
    discharge_times = np.array([admit_times[i] + timedelta(days=int(los_days[i])) for i in range(admissions_n)])

    # Acuity proxy at admission (latent), influenced by age/comorbidities
    pat_lookup = patients.set_index("patient_id")
    age_adm = pat_lookup.loc[adm_patient, "age"].to_numpy()
    com_cnt = pat_lookup.loc[adm_patient, "comorbidity_count"].to_numpy()

    acuity = (0.02 * (age_adm - 50) + 0.25 * com_cnt + rng.normal(0, 0.8, size=admissions_n))
    # Keep in a reasonable range
    acuity = np.clip(acuity, -2.5, 3.5)

    admissions = pd.DataFrame({
        "admission_id": admission_ids,
        "patient_id": adm_patient,
        "admit_time": admit_times,
        "discharge_time": discharge_times,
        "los_days": los_days.astype(int),
        "admit_acuity": acuity.astype(float),
    })

    # Attach patient attributes (for downstream joins)
    admissions = admissions.merge(patients, on="patient_id", how="left")

    return admissions


def generate_vitals_for_admission(rng: np.random.Generator,
                                 admission_row: pd.Series,
                                 vitals_minutes: int,
                                 miss_rate: float,
                                 noise_std: dict) -> pd.DataFrame:
    """
    Create time series vitals for one admission.
    We model gentle drift; deterioration admissions show worsening trend near end.
    """
    start = admission_row["admit_time"]
    end = admission_row["discharge_time"]
    n_steps = max(1, int(((end - start).total_seconds() / 60) // vitals_minutes))
    times = [start + timedelta(minutes=vitals_minutes * i) for i in range(n_steps + 1)]

    acuity = float(admission_row["admit_acuity"])
    age = int(admission_row["age"])
    com_cnt = int(admission_row["comorbidity_count"])

    # Baselines (rough)
    hr_base = 78 + 3.0 * acuity + 0.08 * (age - 50) + 1.5 * com_cnt
    sbp_base = 122 - 4.0 * acuity - 0.10 * (age - 50) - 2.0 * com_cnt
    dbp_base = 74 - 2.0 * acuity - 0.06 * (age - 50) - 1.0 * com_cnt
    spo2_base = 97.0 - 0.6 * acuity - 0.3 * com_cnt

    # Random walk-ish drift
    t = np.linspace(0, 1, len(times))
    hr = hr_base + 6 * (t - 0.5) + rng.normal(0, noise_std["hr"], size=len(times))
    sbp = sbp_base - 8 * (t - 0.5) + rng.normal(0, noise_std["sbp"], size=len(times))
    dbp = dbp_base - 5 * (t - 0.5) + rng.normal(0, noise_std["dbp"], size=len(times))
    spo2 = spo2_base - 1.2 * (t - 0.5) + rng.normal(0, noise_std["spo2"], size=len(times))

    df = pd.DataFrame({
        "admission_id": admission_row["admission_id"],
        "patient_id": admission_row["patient_id"],
        "event_time": times,
        "hr": hr,
        "sbp": sbp,
        "dbp": dbp,
        "spo2": spo2,
    })

    # Missingness: drop individual measurements per variable
    for col in ["hr", "sbp", "dbp", "spo2"]:
        mask = rng.random(len(df)) < miss_rate
        df.loc[mask, col] = np.nan

    return df


def generate_labs_for_admission(rng: np.random.Generator,
                               admission_row: pd.Series,
                               labs_hours: int,
                               miss_rate: float,
                               noise_std: dict) -> pd.DataFrame:
    start = admission_row["admit_time"]
    end = admission_row["discharge_time"]
    n_steps = max(1, int(((end - start).total_seconds() / 3600) // labs_hours))
    times = [start + timedelta(hours=labs_hours * i) for i in range(n_steps + 1)]

    acuity = float(admission_row["admit_acuity"])
    com_cnt = int(admission_row["comorbidity_count"])

    # baselines
    wbc_base = 7.5 + 0.9 * acuity + 0.4 * com_cnt
    cr_base = 0.95 + 0.12 * acuity + 0.18 * (1 if admission_row["ckd"] == 1 else 0)
    lac_base = 1.2 + 0.35 * acuity + 0.12 * com_cnt

    t = np.linspace(0, 1, len(times))
    wbc = wbc_base + 1.0 * (t - 0.5) + rng.normal(0, noise_std["wbc"], size=len(times))
    creat = cr_base + 0.10 * (t - 0.5) + rng.normal(0, noise_std["creatinine"], size=len(times))
    lact = lac_base + 0.35 * (t - 0.5) + rng.normal(0, noise_std["lactate"], size=len(times))

    df = pd.DataFrame({
        "admission_id": admission_row["admission_id"],
        "patient_id": admission_row["patient_id"],
        "event_time": times,
        "wbc": wbc,
        "creatinine": creat,
        "lactate": lact,
    })

    for col in ["wbc", "creatinine", "lactate"]:
        mask = rng.random(len(df)) < miss_rate
        df.loc[mask, col] = np.nan

    return df


def assign_outcomes(rng: np.random.Generator, admissions: pd.DataFrame, synth_cfg: dict) -> pd.DataFrame:
    """
    Create a deterioration event time for a subset of admissions.
    We make probability depend on acuity, age, comorbidities.
    """
    rate = float(synth_cfg["outcomes"]["deterioration_rate"])
    hmin = int(synth_cfg["outcomes"]["label_horizon_hours_min"])
    hmax = int(synth_cfg["outcomes"]["label_horizon_hours_max"])

    acuity = admissions["admit_acuity"].to_numpy()
    age = admissions["age"].to_numpy()
    com_cnt = admissions["comorbidity_count"].to_numpy()

    # Convert desired base rate to intercept; then add covariates
    # Start with logit(rate) and let covariates shift around it.
    base_logit = math.log(rate / (1 - rate))
    logit = base_logit + 0.55 * acuity + 0.02 * (age - 55) + 0.30 * com_cnt + rng.normal(0, 0.35, size=len(admissions))
    p = sigmoid(logit)

    y = rng.binomial(1, p)
    # ensure not insane: keep prevalence close-ish to requested
    # (optional: not forcing exact)
    outcome = y.astype(int)

    # Event time: near the end but not after discharge.
    # For positives, event occurs between (discharge - 48h) and (discharge - 6h), bounded by stay length.
    event_times = []
    for i, row in admissions.iterrows():
        if outcome[i] == 0:
            event_times.append(pd.NaT)
            continue

        admit = row["admit_time"]
        discharge = row["discharge_time"]
        stay_hours = max(1, int((discharge - admit).total_seconds() / 3600))

        # prefer late deterioration
        latest = discharge - timedelta(hours=6)
        earliest = max(admit + timedelta(hours=hmin), discharge - timedelta(hours=hmax))
        if earliest >= latest:
            # fallback: middle of stay
            earliest = admit + timedelta(hours=max(6, stay_hours // 2))
            latest = discharge - timedelta(hours=3)

        # sample uniformly in [earliest, latest]
        span = int((latest - earliest).total_seconds())
        if span <= 0:
            t = earliest
        else:
            t = earliest + timedelta(seconds=int(rng.integers(0, span)))
        event_times.append(t)

    outcomes = pd.DataFrame({
        "admission_id": admissions["admission_id"].values,
        "patient_id": admissions["patient_id"].values,
        "deterioration": outcome,
        "deterioration_time": event_times,
        "outcome_type": np.where(outcome == 1, "deterioration", "none")
    })
    return outcomes


def add_event_date(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df["event_date"] = pd.to_datetime(df[ts_col]).dt.date
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_cfg", default="configs/project.yml")
    parser.add_argument("--synth_cfg", default="configs/synth.yml")
    args = parser.parse_args()

    project_cfg = load_yaml(args.project_cfg)
    synth_cfg = load_yaml(args.synth_cfg)

    # Respect deterministic seed
    seed = int(project_cfg["project"].get("seed", synth_cfg.get("seed", 42)))
    rng = np.random.default_rng(seed)

    spark = build_spark(project_cfg)

    raw_dir = project_cfg["paths"]["raw"]
    ensure_dir(raw_dir)

    # 1) Admissions (pandas -> spark)
    admissions_pd = synthesize_admissions(rng, synth_cfg)

    # 2) Outcomes
    outcomes_pd = assign_outcomes(rng, admissions_pd, synth_cfg)

    # 3) Generate vitals/labs in chunks (avoid huge memory spikes)
    vitals_minutes = int(synth_cfg["cadence"]["vitals_minutes"])
    labs_hours = int(synth_cfg["cadence"]["labs_hours"])
    vit_miss = float(synth_cfg["missingness"]["vitals_missing_rate"])
    lab_miss = float(synth_cfg["missingness"]["labs_missing_rate"])

    v_noise = synth_cfg["noise"]["vitals_noise_std"]
    l_noise = synth_cfg["noise"]["labs_noise_std"]

    # For memory: generate in batches of admissions
    batch_size = 250  # safe for 6GB RAM
    vitals_batches = []
    labs_batches = []

    # Make a quick lookup for deterioration admissions to slightly worsen their trends near the end
    det_map = outcomes_pd.set_index("admission_id")["deterioration"].to_dict()

    for start_idx in range(0, len(admissions_pd), batch_size):
        batch = admissions_pd.iloc[start_idx:start_idx + batch_size]

        v_list = []
        l_list = []
        for _, row in batch.iterrows():
            vdf = generate_vitals_for_admission(rng, row, vitals_minutes, vit_miss, v_noise)
            ldf = generate_labs_for_admission(rng, row, labs_hours, lab_miss, l_noise)

            # If deterioration, nudge last part worse (subtle but learnable)
            if det_map.get(row["admission_id"], 0) == 1:
                # last 20% of timeline: hr up, sbp down, spo2 down, lactate up
                n = len(vdf)
                cut = int(n * 0.8)
                vdf.loc[cut:, "hr"] += 10 + rng.normal(0, 2, size=n - cut)
                vdf.loc[cut:, "sbp"] -= 8 + rng.normal(0, 2, size=n - cut)
                vdf.loc[cut:, "spo2"] -= 1.5 + rng.normal(0, 0.5, size=n - cut)

                m = len(ldf)
                cut2 = int(m * 0.8)
                ldf.loc[cut2:, "lactate"] += 1.2 + rng.normal(0, 0.3, size=m - cut2)
                ldf.loc[cut2:, "wbc"] += 1.0 + rng.normal(0, 0.4, size=m - cut2)

            v_list.append(vdf)
            l_list.append(ldf)

        vitals_pd = pd.concat(v_list, ignore_index=True)
        labs_pd = pd.concat(l_list, ignore_index=True)

        vitals_pd = add_event_date(vitals_pd, "event_time")
        labs_pd = add_event_date(labs_pd, "event_time")

        vitals_batches.append(vitals_pd)
        labs_batches.append(labs_pd)

        # free memory in loop
        del vitals_pd, labs_pd, v_list, l_list

    vitals_all = pd.concat(vitals_batches, ignore_index=True)
    labs_all = pd.concat(labs_batches, ignore_index=True)

    # 4) Convert to Spark DataFrames (explicit schemas)
    admissions_schema = StructType([
        StructField("admission_id", StringType(), False),
        StructField("patient_id", StringType(), False),
        StructField("admit_time", TimestampType(), False),
        StructField("discharge_time", TimestampType(), False),
        StructField("los_days", IntegerType(), False),
        StructField("admit_acuity", DoubleType(), False),
        StructField("age", IntegerType(), False),
        StructField("sex", StringType(), False),
        StructField("diabetes", IntegerType(), False),
        StructField("ckd", IntegerType(), False),
        StructField("copd", IntegerType(), False),
        StructField("chf", IntegerType(), False),
        StructField("comorbidity_count", IntegerType(), False),
    ])

    vitals_schema = StructType([
        StructField("admission_id", StringType(), False),
        StructField("patient_id", StringType(), False),
        StructField("event_time", TimestampType(), False),
        StructField("hr", DoubleType(), True),
        StructField("sbp", DoubleType(), True),
        StructField("dbp", DoubleType(), True),
        StructField("spo2", DoubleType(), True),
        StructField("event_date", DateType(), False),
    ])

    labs_schema = StructType([
        StructField("admission_id", StringType(), False),
        StructField("patient_id", StringType(), False),
        StructField("event_time", TimestampType(), False),
        StructField("wbc", DoubleType(), True),
        StructField("creatinine", DoubleType(), True),
        StructField("lactate", DoubleType(), True),
        StructField("event_date", DateType(), False),
    ])

    outcomes_schema = StructType([
        StructField("admission_id", StringType(), False),
        StructField("patient_id", StringType(), False),
        StructField("deterioration", IntegerType(), False),
        StructField("deterioration_time", TimestampType(), True),
        StructField("outcome_type", StringType(), False),
    ])

    adm_sdf = spark.createDataFrame(admissions_pd, schema=admissions_schema)
    vit_sdf = spark.createDataFrame(vitals_all, schema=vitals_schema)
    lab_sdf = spark.createDataFrame(labs_all, schema=labs_schema)
    out_sdf = spark.createDataFrame(outcomes_pd, schema=outcomes_schema)

    # 5) Write raw tables
    raw_format = synth_cfg["storage"].get("raw_format", "parquet")
    part_cols = synth_cfg["storage"].get("partition_cols", ["event_date"])

    def abs_path(p: str) -> str:
        return os.path.abspath(p)

    adm_path = abs_path(os.path.join(raw_dir, "admissions"))
    out_path = abs_path(os.path.join(raw_dir, "outcomes"))
    vit_path = abs_path(os.path.join(raw_dir, "vitals"))
    lab_path = abs_path(os.path.join(raw_dir, "labs"))

    # admissions / outcomes: not partitioned (small)
    (adm_sdf
     .repartition(8)
     .write.mode("overwrite").format(raw_format).save(adm_path))

    (out_sdf
     .repartition(8)
     .write.mode("overwrite").format(raw_format).save(out_path))

    # vitals / labs: partition by event_date for faster windowing later
    (vit_sdf
     .repartition(16, F.col("event_date"))
     .write.mode("overwrite").partitionBy(*part_cols).format(raw_format).save(vit_path))

    (lab_sdf
     .repartition(16, F.col("event_date"))
     .write.mode("overwrite").partitionBy(*part_cols).format(raw_format).save(lab_path))

    # 6) Print summary (light, helpful)
    print("✅ Synthetic raw data generated")
    print("Admissions:", adm_sdf.count())
    print("Vitals rows:", vit_sdf.count())
    print("Labs rows:", lab_sdf.count())
    print("Deteriorations:", out_sdf.filter(F.col("deterioration") == 1).count())
    print("Raw output:", raw_dir)

    spark.stop()


if __name__ == "__main__":
    main()