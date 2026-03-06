import os
import json
from datetime import datetime
import yaml

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_spark(project_cfg):
    s = project_cfg["spark"]
    spark = (
        SparkSession.builder
        .master(s["master"])
        .appName(project_cfg["project"]["name"] + "_batch_score")
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def abs_path(p):
    return os.path.abspath(p)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    project_cfg = load_yaml("configs/project.yml")

    reports_dir = abs_path(project_cfg["paths"]["reports"])
    metrics_dir = abs_path(os.path.join(reports_dir, "metrics"))
    gold_dir = abs_path(project_cfg["paths"]["gold"])
    outputs_dir = abs_path(project_cfg["paths"]["outputs"])

    ensure_dir(outputs_dir)
    ensure_dir(os.path.join(outputs_dir, "predictions"))

    spark = build_spark(project_cfg)

    # Load model selection
    eval_path = abs_path(os.path.join(metrics_dir, "eval_summary_latest.json"))
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_summary = json.load(f)

    model_path = eval_summary["model_path"]
    model_name = eval_summary["selected_model"]

    print("Using model:", model_name)
    print("Model path:", model_path)

    # Score ALL rows (train+val+test) so we can simulate a batch run
    ds = spark.read.parquet(abs_path(os.path.join(gold_dir, "training_set")))

    # Load model and score
    model = PipelineModel.load(model_path)
    pred = model.transform(ds).withColumn("risk_score", vector_to_array("probability")[1])

    # Choose final columns (hospital-friendly)
    out = pred.select(
        "patient_id", "admission_id", "anchor_time", "anchor_date",
        "risk_score", "label", "split"
    )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = abs_path(os.path.join(outputs_dir, "predictions", f"risk_scores_{run_id}"))

    out.write.mode("overwrite").parquet(out_path)

    # Also write a small CSV sample for quick viewing
    sample_path = abs_path(os.path.join(outputs_dir, "predictions", f"risk_scores_sample_{run_id}"))
    (
        out.orderBy(F.col("risk_score").desc())
        .limit(200)
        .coalesce(1)
        .write.mode("overwrite").option("header", True).csv(sample_path)
    )

    print("✅ Batch scoring complete")
    print("Parquet output:", out_path)
    print("CSV sample folder:", sample_path)

    spark.stop()


if __name__ == "__main__":
    main()