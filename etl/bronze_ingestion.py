from __future__ import annotations

import sys
from typing import Optional

from pyspark.sql import SparkSession, DataFrame, functions as F

from utils.config import RunConfig, ensure_dirs, raw_table_path, layer_table_path
from utils.validation_checks import (
    emit,
    emit_many,
    row_count,
    bronze_checks_patients,
    bronze_checks_vitals,
    bronze_checks_labs,
    bronze_checks_outcomes,
    fk_admission_exists,
    event_within_admission_window,
    QualityCheckError,
)


# -----------------------------
# Spark
# -----------------------------

def create_spark(app_name: str = "clinical-deterioration-bronze") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
    return spark


# -----------------------------
# I/O
# -----------------------------

def read_parquet_dir(spark: SparkSession, path: str) -> DataFrame:
    """
    Spark reads Parquet from a directory. This supports multiple files under the folder.
    """
    return spark.read.parquet(path)


def write_parquet_dir(df: DataFrame, out_path: str, partition_col: Optional[str] = None) -> None:
    writer = df.write.mode("overwrite").format("parquet")
    if partition_col and partition_col in df.columns:
        writer = writer.partitionBy(partition_col)
    writer.save(out_path)

def normalize_patients_minimal(df: DataFrame, run_id: str) -> DataFrame:
    """
    Bronze intent: preserve structure, ensure parseable timestamps, stable types.
    """
    out = df

    # Parse timestamps into *_ts columns (keep originals)
    if "admission_time_ts" not in out.columns:
        if "admission_time" not in out.columns:
            # allow QC to catch missing required columns; still create column to avoid crashing later
            out = out.withColumn("admission_time_ts", F.lit(None).cast("timestamp"))
        else:
            out = out.withColumn("admission_time_ts", F.to_timestamp(F.col("admission_time")))

    if "discharge_time_ts" not in out.columns:
        if "discharge_time" in out.columns:
            out = out.withColumn("discharge_time_ts", F.to_timestamp(F.col("discharge_time")))
        else:
            out = out.withColumn("discharge_time_ts", F.lit(None).cast("timestamp"))

    # Basic casts
    if "age" in out.columns:
        out = out.withColumn("age", F.col("age").cast("int"))

    # Lineage
    if "ingest_run_id" not in out.columns:
        out = out.withColumn("ingest_run_id", F.lit(run_id))
    else:
        out = out.withColumn("ingest_run_id", F.lit(run_id))

    return out


def normalize_events_minimal(df: DataFrame, run_id: str, time_col: str = "event_time") -> DataFrame:
    """
    For vitals_events and labs_events. Creates event_time_ts and casts numeric columns if present.
    """
    out = df

    if "event_time_ts" not in out.columns:
        if time_col not in out.columns:
            out = out.withColumn("event_time_ts", F.lit(None).cast("timestamp"))
        else:
            out = out.withColumn("event_time_ts", F.to_timestamp(F.col(time_col)))

    # Cast likely numeric columns if present (safe, no imputation)
    numeric_cols = [
        "hr", "sbp", "dbp", "rr", "temp_c", "spo2",
        "wbc", "creatinine", "lactate",
        "sodium", "potassium", "bun"
    ]
    for c in numeric_cols:
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast("double"))

    if "ingest_run_id" not in out.columns:
        out = out.withColumn("ingest_run_id", F.lit(run_id))
    else:
        out = out.withColumn("ingest_run_id", F.lit(run_id))

    return out


def normalize_outcomes_minimal(df: DataFrame, run_id: str) -> DataFrame:
    """
    Outcomes: creates outcome_time_ts, ensures outcome_flag exists and is int.
    """
    out = df

    if "outcome_time_ts" not in out.columns:
        if "outcome_time" not in out.columns:
            out = out.withColumn("outcome_time_ts", F.lit(None).cast("timestamp"))
        else:
            out = out.withColumn("outcome_time_ts", F.to_timestamp(F.col("outcome_time")))

    if "outcome_flag" in out.columns:
        out = out.withColumn("outcome_flag", F.col("outcome_flag").cast("int"))
    else:
        out = out.withColumn("outcome_flag", F.lit(1).cast("int"))

    if "ingest_run_id" not in out.columns:
        out = out.withColumn("ingest_run_id", F.lit(run_id))
    else:
        out = out.withColumn("ingest_run_id", F.lit(run_id))

    return out


# -----------------------------
# Bronze pipeline
# -----------------------------

def run_bronze(cfg: RunConfig) -> None:
    ensure_dirs(cfg)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[INFO] Bronze run_id={cfg.run_id}")
    print(f"[INFO] Raw format locked: {cfg.raw_format}")
    print(f"[INFO] Raw dir: {cfg.paths.raw_dir}")
    print(f"[INFO] Bronze dir: {cfg.paths.bronze_dir}")

    # ---- Read RAW Parquet ----
    patients_raw_path = raw_table_path(cfg, "patients")
    vitals_raw_path = raw_table_path(cfg, "vitals_events")
    labs_raw_path = raw_table_path(cfg, "labs_events")
    outcomes_raw_path = raw_table_path(cfg, "outcomes")

    patients_raw = read_parquet_dir(spark, str(patients_raw_path))
    vitals_raw = read_parquet_dir(spark, str(vitals_raw_path))
    labs_raw = read_parquet_dir(spark, str(labs_raw_path))
    outcomes_raw = read_parquet_dir(spark, str(outcomes_raw_path))

    # ---- Minimal normalization ----
    patients = normalize_patients_minimal(patients_raw, cfg.run_id)
    vitals = normalize_events_minimal(vitals_raw, cfg.run_id, time_col="event_time")
    labs = normalize_events_minimal(labs_raw, cfg.run_id, time_col="event_time")
    outcomes = normalize_outcomes_minimal(outcomes_raw, cfg.run_id)

    # ---- Bronze QC (fail fast on ERROR) ----
    # Basic row counts
    emit(row_count(patients, "patients_norm"))
    emit(row_count(vitals, "vitals_norm"))
    emit(row_count(labs, "labs_norm"))
    emit(row_count(outcomes, "outcomes_norm"))

    # Table-level checks
    emit_many(bronze_checks_patients(patients))
    emit_many(bronze_checks_vitals(vitals))
    emit_many(bronze_checks_labs(labs))
    emit_many(bronze_checks_outcomes(outcomes))

    # Relational + temporal checks (ERROR)
    emit(fk_admission_exists(vitals, patients, "vitals"))
    emit(event_within_admission_window(vitals, patients, "event_time_ts", "vitals"))

    emit(fk_admission_exists(labs, patients, "labs"))
    emit(event_within_admission_window(labs, patients, "event_time_ts", "labs"))

    # outcomes: reuse window check by renaming outcome_time_ts -> event_time_ts
    outcomes_for_window = outcomes.withColumnRenamed("outcome_time_ts", "event_time_ts")
    emit(fk_admission_exists(outcomes_for_window, patients, "outcomes"))
    emit(event_within_admission_window(outcomes_for_window, patients, "event_time_ts", "outcomes"))

    # ---- Write Bronze Parquet ----
    patients_out = layer_table_path(cfg, "bronze", "patients_bronze")
    vitals_out = layer_table_path(cfg, "bronze", "vitals_events_bronze")
    labs_out = layer_table_path(cfg, "bronze", "labs_events_bronze")
    outcomes_out = layer_table_path(cfg, "bronze", "outcomes_bronze")

    write_parquet_dir(patients, str(patients_out), cfg.partition_col)
    write_parquet_dir(vitals, str(vitals_out), cfg.partition_col)
    write_parquet_dir(labs, str(labs_out), cfg.partition_col)
    write_parquet_dir(outcomes, str(outcomes_out), cfg.partition_col)

    print("[INFO] Bronze ingestion completed successfully.")
    spark.stop()


def main() -> None:
    cfg = RunConfig.default()
    try:
        run_bronze(cfg)
    except QualityCheckError as e:
        print(f"[FATAL] Bronze ingestion failed quality checks: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Bronze ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
