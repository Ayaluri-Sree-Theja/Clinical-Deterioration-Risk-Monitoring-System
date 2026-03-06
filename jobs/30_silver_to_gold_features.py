import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# -----------------------------
# Helpers
# -----------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_spark(project_cfg):

    s = project_cfg["spark"]

    spark = (
        SparkSession.builder
        .master(s["master"])
        .appName(project_cfg["project"]["name"] + "_gold_features")
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def abs_path(p):
    return os.path.abspath(p)


# -----------------------------
# MAIN
# -----------------------------
def main():

    project_cfg = load_yaml("configs/project.yml")
    feature_cfg = load_yaml("configs/features.yml")

    spark = build_spark(project_cfg)

    silver_dir = project_cfg["paths"]["silver"]
    gold_dir = project_cfg["paths"]["gold"]

    os.makedirs(gold_dir, exist_ok=True)

    cadence_h = int(project_cfg["time"]["anchor_cadence_hours"])
    windows = feature_cfg["windows_hours"]

    # -------------------------
    # Load SILVER tables
    # -------------------------
    admissions = spark.read.parquet(abs_path(os.path.join(silver_dir, "admissions")))
    vitals = spark.read.parquet(abs_path(os.path.join(silver_dir, "vitals")))
    labs = spark.read.parquet(abs_path(os.path.join(silver_dir, "labs")))

    # -------------------------
    # Build anchors
    # -------------------------
    anchors = (
        admissions
        .withColumn(
            "anchor_time",
            F.explode(
                F.sequence(
                    F.col("admit_time"),
                    F.expr(f"discharge_time - interval {cadence_h} hours"),
                    F.expr(f"interval {cadence_h} hours")
                )
            )
        )
        .select("patient_id", "admission_id", "anchor_time")
        .withColumn("anchor_date", F.to_date("anchor_time"))
    )

    anchors_path = abs_path(os.path.join(gold_dir, "anchors"))
    anchors.write.mode("overwrite").parquet(anchors_path)

    print("Anchors built:", anchors.count())

    # -------------------------
    # FEATURE WINDOWS
    # -------------------------
    feature_dfs = []

    for w in windows:

        print(f"Building window features: {w}h")

        vitals_w = (
            anchors.join(
                vitals,
                ["patient_id", "admission_id"],
                "left"
            )
            .where(
                (F.col("event_time") <= F.col("anchor_time")) &
                (F.col("event_time") > F.col("anchor_time") - F.expr(f"interval {w} hours"))
            )
            .groupBy("patient_id", "admission_id", "anchor_time")
            .agg(
                F.mean("hr").alias(f"hr_mean_{w}h"),
                F.min("spo2").alias(f"spo2_min_{w}h"),
                F.last("hr", ignorenulls=True).alias(f"hr_last_{w}h"),
                F.count("hr").alias(f"hr_count_{w}h")
            )
        )

        labs_w = (
            anchors.join(
                labs,
                ["patient_id", "admission_id"],
                "left"
            )
            .where(
                (F.col("event_time") <= F.col("anchor_time")) &
                (F.col("event_time") > F.col("anchor_time") - F.expr(f"interval {w} hours"))
            )
            .groupBy("patient_id", "admission_id", "anchor_time")
            .agg(
                F.mean("lactate").alias(f"lactate_mean_{w}h"),
                F.last("creatinine", ignorenulls=True).alias(f"creatinine_last_{w}h"),
                F.count("lactate").alias(f"lactate_count_{w}h")
            )
        )

        feature_dfs.append(vitals_w.join(labs_w,
                                         ["patient_id", "admission_id", "anchor_time"],
                                         "outer"))

    # -------------------------
    # Merge all window features
    # -------------------------
    features = anchors

    for fdf in feature_dfs:
        features = features.join(
            fdf,
            ["patient_id", "admission_id", "anchor_time"],
            "left"
        )

    features_path = abs_path(os.path.join(gold_dir, "features"))

    (
        features
        .repartition(16)
        .write
        .mode("overwrite")
        .parquet(features_path)
    )

    print("✅ Gold features created")
    print("Feature rows:", features.count())

    spark.stop()


if __name__ == "__main__":
    main()