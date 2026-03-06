import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_spark(project_cfg):
    s = project_cfg["spark"]
    spark = (
        SparkSession.builder
        .master(s["master"])
        .appName(project_cfg["project"]["name"] + "_training_set")
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def abs_path(p):
    return os.path.abspath(p)


def main():

    project_cfg = load_yaml("configs/project.yml")

    silver_dir = project_cfg["paths"]["silver"]
    gold_dir = project_cfg["paths"]["gold"]

    spark = build_spark(project_cfg)

    # Load gold
    features = spark.read.parquet(abs_path(os.path.join(gold_dir, "features")))
    labels = spark.read.parquet(abs_path(os.path.join(gold_dir, "labels"))).select(
        "patient_id", "admission_id", "anchor_time", "label"
    )

    # Bring in static admission attributes (safe: no future info)
    admissions = spark.read.parquet(abs_path(os.path.join(silver_dir, "admissions"))).select(
        "patient_id", "admission_id",
        "age", "sex",
        "diabetes", "ckd", "copd", "chf",
        "comorbidity_count",
        "admit_acuity"
    )

    # Join into one modeling table
    ds = (
        features.join(labels, ["patient_id", "admission_id", "anchor_time"], "inner")
        .join(admissions, ["patient_id", "admission_id"], "left")
    )

    # Fill missing numeric features with null-safe defaults (keep it simple for Spark ML)
    # Note: We do NOT fill label.
    numeric_cols = [c for c, t in ds.dtypes if t in ("double", "int", "bigint") and c != "label"]
    for c in numeric_cols:
        ds = ds.withColumn(c, F.when(F.col(c).isNull(), F.lit(0.0)).otherwise(F.col(c)))

    # Encode sex as binary (Spark ML likes numeric)
    ds = ds.withColumn("sex_m", F.when(F.col("sex") == "M", F.lit(1)).otherwise(F.lit(0))).drop("sex")

    # Add split key (patient-level split to avoid leakage)
    # hash() is deterministic within Spark; use pmod for stable buckets.
    ds = ds.withColumn("split_bucket", F.pmod(F.abs(F.hash(F.col("patient_id"))), F.lit(100)))

    train_frac = float(project_cfg["splits"]["train_frac"])
    val_frac = float(project_cfg["splits"]["val_frac"])

    train_cut = int(train_frac * 100)                 # e.g., 70
    val_cut = int((train_frac + val_frac) * 100)      # e.g., 85

    ds = (
        ds.withColumn(
            "split",
            F.when(F.col("split_bucket") < train_cut, F.lit("train"))
             .when(F.col("split_bucket") < val_cut, F.lit("val"))
             .otherwise(F.lit("test"))
        )
        .drop("split_bucket")
    )

    out_path = abs_path(os.path.join(gold_dir, "training_set"))
    ds.write.mode("overwrite").parquet(out_path)

    print("✅ Gold training_set created")
    print("Rows:", ds.count())
    print("Train:", ds.filter(F.col("split") == "train").count())
    print("Val:", ds.filter(F.col("split") == "val").count())
    print("Test:", ds.filter(F.col("split") == "test").count())
    print("Positives (overall):", ds.filter(F.col("label") == 1).count())

    spark.stop()


if __name__ == "__main__":
    main()