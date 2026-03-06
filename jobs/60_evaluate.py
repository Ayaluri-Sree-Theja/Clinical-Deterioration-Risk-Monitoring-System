import os
import json
import glob
import yaml

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_spark(project_cfg):
    s = project_cfg["spark"]
    spark = (
        SparkSession.builder
        .master(s["master"])
        .appName(project_cfg["project"]["name"] + "_evaluate")
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def abs_path(p):
    return os.path.abspath(p)

def latest_metrics_file(metrics_dir):
    files = sorted(glob.glob(os.path.join(metrics_dir, "train_metrics_*.json")))
    if not files:
        raise FileNotFoundError("No train_metrics_*.json found in reports/metrics/")
    return files[-1]

def pick_best_model(metrics_json):
    # choose best by test auprc
    best_name = None
    best_auprc = -1
    for name, info in metrics_json["models"].items():
        auprc = info["test"]["metrics"]["auprc"]
        if auprc > best_auprc:
            best_auprc = auprc
            best_name = name
    return best_name, best_auprc

def evaluate(pred_df, label_col="label"):
    auroc_eval = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auprc_eval = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    return float(auroc_eval.evaluate(pred_df)), float(auprc_eval.evaluate(pred_df))

def topk(pred_df, k_rate, label_col="label"):
    n = pred_df.count()
    k = int(n * k_rate)
    if k < 1:
        return {"k_rate": k_rate, "k": 0}

    total_pos = pred_df.filter(F.col(label_col) == 1).count()
    base = total_pos / n if n > 0 else 0.0

    top = pred_df.orderBy(F.col("prob").desc()).limit(k)
    top_pos = top.filter(F.col(label_col) == 1).count()

    precision = top_pos / k
    recall = top_pos / total_pos if total_pos > 0 else 0.0
    lift = (precision / base) if base > 0 else None

    return {
        "k_rate": k_rate,
        "k": k,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
        "base_rate": base
    }

def calibration_table(pred_df, bins=10):
    # equal-width bins on probability
    cal = (
        pred_df
        .withColumn("bin", F.floor(F.col("prob") * F.lit(bins)))
        .withColumn("bin", F.when(F.col("bin") == bins, bins - 1).otherwise(F.col("bin")))
        .groupBy("bin")
        .agg(
            F.count("*").alias("n"),
            F.avg("prob").alias("avg_pred"),
            F.avg("label").alias("event_rate")
        )
        .orderBy("bin")
    )
    return cal

def main():
    project_cfg = load_yaml("configs/project.yml")

    reports_dir = project_cfg["paths"]["reports"]
    metrics_dir = abs_path(os.path.join(reports_dir, "metrics"))
    gold_dir = project_cfg["paths"]["gold"]

    spark = build_spark(project_cfg)

    # Load latest training metrics
    mfile = latest_metrics_file(metrics_dir)
    with open(mfile, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    best_model, best_auprc = pick_best_model(metrics)
    model_path = metrics["models"][best_model]["model_path"]

    print("Latest metrics file:", mfile)
    print("Best model:", best_model, "Best TEST AUPRC:", best_auprc)
    print("Model path:", model_path)

    # Load dataset + filter test
    ds = spark.read.parquet(abs_path(os.path.join(gold_dir, "training_set")))
    test = ds.filter(F.col("split") == "test").drop("split")

    # Load fitted pipeline model
    model = PipelineModel.load(model_path)

    pred = model.transform(test)
    pred = pred.withColumn("prob", vector_to_array("probability")[1])

    auroc, auprc = evaluate(pred)
    summary = {
        "selected_model": best_model,
        "model_path": model_path,
        "test_auroc": auroc,
        "test_auprc": auprc,
        "topk": [
            topk(pred.select("label", "prob", "rawPrediction"), 0.05),
            topk(pred.select("label", "prob", "rawPrediction"), 0.10),
            topk(pred.select("label", "prob", "rawPrediction"), 0.20),
        ]
    }

    # Save summary json
    out_json = abs_path(os.path.join(metrics_dir, "eval_summary_latest.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save calibration csv
    cal = calibration_table(pred.select("label", "prob"), bins=10)
    out_cal = abs_path(os.path.join(metrics_dir, "calibration_test"))
    cal.coalesce(1).write.mode("overwrite").option("header", True).csv(out_cal)

    print("\n✅ Evaluation complete")
    print("Saved:", out_json)
    print("Saved calibration folder:", out_cal)

    spark.stop()

if __name__ == "__main__":
    main()