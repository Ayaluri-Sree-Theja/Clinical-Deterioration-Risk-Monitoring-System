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
        .appName(project_cfg["project"]["name"] + "_gold_labels")
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

    spark = build_spark(project_cfg)

    silver_dir = project_cfg["paths"]["silver"]
    gold_dir = project_cfg["paths"]["gold"]

    hmin = int(project_cfg["time"]["label_horizon_hours_min"])
    hmax = int(project_cfg["time"]["label_horizon_hours_max"])

    anchors = spark.read.parquet(abs_path(os.path.join(gold_dir, "anchors"))).select(
        "patient_id", "admission_id", "anchor_time", "anchor_date"
    )

    outcomes = spark.read.parquet(abs_path(os.path.join(silver_dir, "outcomes"))).select(
        "patient_id", "admission_id", "deterioration", "deterioration_time"
    )

    labeled = (
        anchors.join(outcomes, ["patient_id", "admission_id"], "left")
        .withColumn("label_window_start", F.col("anchor_time") + F.expr(f"interval {hmin} hours"))
        .withColumn("label_window_end", F.col("anchor_time") + F.expr(f"interval {hmax} hours"))
        .withColumn(
            "label",
            F.when(
                (F.col("deterioration") == 1) &
                (F.col("deterioration_time").isNotNull()) &
                (F.col("deterioration_time") > F.col("label_window_start")) &
                (F.col("deterioration_time") <= F.col("label_window_end")),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .select(
            "patient_id", "admission_id", "anchor_time", "anchor_date",
            "label", "deterioration_time"
        )
    )

    labels_path = abs_path(os.path.join(gold_dir, "labels"))
    labeled.write.mode("overwrite").parquet(labels_path)

    print("✅ Gold labels created")
    print("Label rows:", labeled.count())
    print("Positive labels:", labeled.filter(F.col("label") == 1).count())

    spark.stop()


if __name__ == "__main__":
    main()