import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_spark(project_cfg):

    spark_cfg = project_cfg["spark"]

    spark = (
        SparkSession.builder
        .master(spark_cfg["master"])
        .appName(project_cfg["project"]["name"] + "_silver")
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(spark_cfg["shuffle_partitions"]))
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def abs_path(p):
    return os.path.abspath(p)


def clip_column(df, col, lo, hi):
    return df.withColumn(col, F.when(F.col(col) < lo, lo)
                              .when(F.col(col) > hi, hi)
                              .otherwise(F.col(col)))


def main():

    project_cfg = load_yaml("configs/project.yml")
    feature_cfg = load_yaml("configs/features.yml")

    spark = build_spark(project_cfg)

    bronze_dir = project_cfg["paths"]["bronze"]
    silver_dir = project_cfg["paths"]["silver"]

    os.makedirs(silver_dir, exist_ok=True)

    # ----------------------------
    # Admissions
    # ----------------------------
    print("\nCleaning admissions...")

    admissions = spark.read.parquet(abs_path(os.path.join(bronze_dir, "admissions")))

    admissions = (
        admissions
        .dropDuplicates(["patient_id", "admission_id"])
    )

    admissions.write.mode("overwrite").parquet(abs_path(os.path.join(silver_dir, "admissions")))

    print("Admissions cleaned")

    # ----------------------------
    # Vitals cleaning
    # ----------------------------
    print("\nCleaning vitals...")

    vitals = spark.read.parquet(abs_path(os.path.join(bronze_dir, "vitals")))

    clip_ranges = feature_cfg["normalization"]["clip_ranges"]

    for col in ["hr", "sbp", "dbp", "spo2"]:
        lo, hi = clip_ranges[col]
        vitals = clip_column(vitals, col, lo, hi)

    vitals = vitals.dropDuplicates(["patient_id", "admission_id", "event_time"])

    vitals.write.mode("overwrite").parquet(abs_path(os.path.join(silver_dir, "vitals")))

    print("Vitals cleaned")

    # ----------------------------
    # Labs cleaning
    # ----------------------------
    print("\nCleaning labs...")

    labs = spark.read.parquet(abs_path(os.path.join(bronze_dir, "labs")))

    for col in ["wbc", "creatinine", "lactate"]:
        lo, hi = clip_ranges[col]
        labs = clip_column(labs, col, lo, hi)

    labs = labs.dropDuplicates(["patient_id", "admission_id", "event_time"])

    labs.write.mode("overwrite").parquet(abs_path(os.path.join(silver_dir, "labs")))

    print("Labs cleaned")

    # ----------------------------
    # Outcomes
    # ----------------------------
    print("\nCleaning outcomes...")

    outcomes = spark.read.parquet(abs_path(os.path.join(bronze_dir, "outcomes")))

    outcomes = outcomes.dropDuplicates(["patient_id", "admission_id"])

    outcomes.write.mode("overwrite").parquet(abs_path(os.path.join(silver_dir, "outcomes")))

    print("Outcomes cleaned")

    print("\n✅ Silver layer ready")
    spark.stop()


if __name__ == "__main__":
    main()