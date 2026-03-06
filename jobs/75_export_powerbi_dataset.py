import os
import glob
import yaml
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_spark(project_cfg):
    s = project_cfg["spark"]
    spark = (
        SparkSession.builder
        .master(s["master"])
        .appName(project_cfg["project"]["name"] + "_export_powerbi")
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def abs_path(p):
    return os.path.abspath(p)


def latest_folder(base_glob):
    paths = sorted(glob.glob(base_glob))
    if not paths:
        raise FileNotFoundError(f"No folders matched: {base_glob}")
    return paths[-1]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    project_cfg = load_yaml("configs/project.yml")
    spark = build_spark(project_cfg)

    outputs_dir = abs_path(project_cfg["paths"]["outputs"])
    dashboard_dir = abs_path(project_cfg["paths"].get("dashboard", os.path.join(outputs_dir, "dashboard")))
    gold_dir = abs_path(project_cfg["paths"]["gold"])

    ensure_dir(dashboard_dir)

    # ---------- Find latest scored parquet folder ----------
    # Prefer real parquet outputs: risk_scores_*
    scored_path = latest_folder(os.path.join(outputs_dir, "predictions", "risk_scores_*"))

    # If the latest match is a *sample* folder (CSV), fallback to latest non-sample parquet
    if "risk_scores_sample_" in scored_path:
        scored_path = latest_folder(os.path.join(outputs_dir, "predictions", "risk_scores_[!s]*"))
    
    print("Using scored parquet:", scored_path)

    scored = spark.read.format("parquet").load(scored_path)

    # scored has: patient_id, admission_id, anchor_time, anchor_date, risk_score, label, split
    # training_set has: all features + label + split etc.
    ts = spark.read.parquet(abs_path(os.path.join(gold_dir, "training_set")))

    # Join minimal-but-useful feature set for tooltips
    tooltip_cols = [
        "age", "sex_m", "comorbidity_count", "admit_acuity", "diabetes", "ckd", "copd", "chf",
        "hr_mean_6h", "hr_last_6h", "hr_count_6h", "spo2_min_6h",
        "lactate_mean_6h", "lactate_count_6h", "creatinine_last_6h",
        "hr_mean_24h", "spo2_min_24h", "lactate_mean_24h", "creatinine_last_24h"
    ]
    # keep only existing columns (in case you adjust features later)
    existing = set(ts.columns)
    tooltip_cols = [c for c in tooltip_cols if c in existing]

    ts_small = ts.select(
        "patient_id", "admission_id", "anchor_time", "anchor_date",
        *tooltip_cols
    )

    df = (
        scored.join(ts_small, ["patient_id", "admission_id", "anchor_time", "anchor_date"], "left")
    )

    # ---------- Risk deciles ----------
    # ntile over probability: decile 10 = highest risk
    w = F.window  # not used; just avoiding lint confusion
    from pyspark.sql.window import Window
    win = Window.orderBy(F.col("risk_score").asc())
    df = df.withColumn("risk_decile", F.ntile(10).over(win))

    # ---------- Alert flag based on alert rate ----------
    # Default alert rate for dashboard: 10%
    alert_rate = float(project_cfg.get("dashboard", {}).get("alert_rate", 0.10))
    # We approximate threshold using quantile to avoid sorting full dataset.
    # threshold = (1 - alert_rate) quantile of risk_score
    q = 1.0 - alert_rate
    thr = df.approxQuantile("risk_score", [q], 0.001)[0]
    df = df.withColumn("alert_flag", F.when(F.col("risk_score") >= F.lit(thr), F.lit(1)).otherwise(F.lit(0)))
    df = df.withColumn("alert_rate_config", F.lit(alert_rate))

    # ---------- Build a 50k dashboard sample ----------
    # Senior trick: mix "alerted" + random background so visuals look real.
    target_n = int(project_cfg.get("dashboard", {}).get("rows", 50000))
    alerted_frac = float(project_cfg.get("dashboard", {}).get("alerted_fraction", 0.50))  # 50% from alerted

    alerted_n = int(target_n * alerted_frac)
    background_n = target_n - alerted_n

    alerted = df.filter(F.col("alert_flag") == 1)
    background = df.filter(F.col("alert_flag") == 0)

    # Sample without collecting: orderBy(rand) is ok for 50k
    alerted_s = alerted.orderBy(F.rand(seed=42)).limit(alerted_n)
    background_s = background.orderBy(F.rand(seed=43)).limit(background_n)

    out = alerted_s.unionByName(background_s)

    # Make time fields Power BI friendly
    out = out.withColumn("anchor_datetime", F.col("anchor_time").cast("timestamp"))
    out = out.drop("anchor_time")  # keep anchor_datetime + anchor_date

    # Optional: human-friendly band
    out = out.withColumn(
        "risk_band",
        F.when(F.col("risk_score") >= 0.80, F.lit("HIGH"))
         .when(F.col("risk_score") >= 0.50, F.lit("MEDIUM"))
         .otherwise(F.lit("LOW"))
    )

    # Export
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = abs_path(os.path.join(dashboard_dir, f"pbi_risk_dataset_{run_id}"))

    (
        out.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(out_path)
    )

    print("✅ Power BI dataset exported")
    print("Rows (exported):", out.count())
    print("Output folder:", out_path)
    print("Threshold used:", thr, "Alert rate:", alert_rate)

    spark.stop()


if __name__ == "__main__":
    main()