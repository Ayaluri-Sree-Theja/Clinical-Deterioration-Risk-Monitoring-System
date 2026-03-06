import os
import json
import yaml
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_spark(project_cfg):
    s = project_cfg["spark"]
    spark = (
        SparkSession.builder
        .master(s["master"])
        .appName(project_cfg["project"]["name"] + "_train_sparkml")
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

def compute_class_weighted_df(df, label_col="label"):
    """
    Add weightCol to handle class imbalance.
    weight = (neg/pos) for positive, 1.0 for negative
    """
    pos = df.filter(F.col(label_col) == 1).count()
    neg = df.filter(F.col(label_col) == 0).count()
    ratio = (neg / pos) if pos > 0 else 1.0

    wdf = df.withColumn(
        "weight",
        F.when(F.col(label_col) == 1, F.lit(float(ratio))).otherwise(F.lit(1.0))
    )
    return wdf, {"pos": pos, "neg": neg, "neg_pos_ratio": ratio}

def topk_metrics(pred_df, k_rate=0.10, label_col="label", score_col="prob"):
    """
    Compute precision/recall at top K% by risk score.
    """
    # total rows
    n = pred_df.count()
    k = int(n * k_rate)

    # if k=0 avoid crash
    if k < 1:
        return {"k_rate": k_rate, "k": 0, "precision_at_k": None, "recall_at_k": None, "lift_at_k": None}

    # baseline prevalence
    total_pos = pred_df.filter(F.col(label_col) == 1).count()
    base_rate = total_pos / n if n > 0 else 0.0

    topk = pred_df.orderBy(F.col(score_col).desc()).limit(k)
    topk_pos = topk.filter(F.col(label_col) == 1).count()

    precision = topk_pos / k
    recall = topk_pos / total_pos if total_pos > 0 else 0.0
    lift = (precision / base_rate) if base_rate > 0 else None

    return {
        "k_rate": k_rate,
        "k": k,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
        "base_rate": base_rate
    }

def evaluate(pred_df, label_col="label"):
    """
    Returns AUROC + AUPRC using Spark evaluator.
    """
    auroc_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auprc_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
    return {
        "auroc": float(auroc_eval.evaluate(pred_df)),
        "auprc": float(auprc_eval.evaluate(pred_df))
    }

def add_weight_col(df, ratio, label_col="label"):
    return df.withColumn(
        "weight",
        F.when(F.col(label_col) == 1, F.lit(float(ratio))).otherwise(F.lit(1.0))
    )

def main():
    project_cfg = load_yaml("configs/project.yml")
    thresh_cfg = load_yaml("configs/thresholds.yml")

    spark = build_spark(project_cfg)

    gold_dir = project_cfg["paths"]["gold"]
    reports_dir = project_cfg["paths"]["reports"]
    ensure_dir(reports_dir)
    ensure_dir(os.path.join(reports_dir, "metrics"))
    ensure_dir(os.path.join(reports_dir, "models"))

    ds = spark.read.parquet(abs_path(os.path.join(gold_dir, "training_set")))

    train = ds.filter(F.col("split") == "train").drop("split")
    val = ds.filter(F.col("split") == "val").drop("split")
    test = ds.filter(F.col("split") == "test").drop("split")

    # compute ratio on train, then apply to all splits
    train_w, class_stats = compute_class_weighted_df(train, label_col="label")
    ratio = class_stats["neg_pos_ratio"]

    val = add_weight_col(val, ratio, label_col="label")
    test = add_weight_col(test, ratio, label_col="label")

    # Assemble features: take all numeric columns except label
    exclude = {"label", "weight", "patient_id", "admission_id", "anchor_time", "anchor_date", "deterioration_time"}
    feature_cols = [c for c, t in train_w.dtypes if (c not in exclude and t in ("double", "int", "bigint"))]
    feature_cols = sorted(feature_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Models
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="weight",
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0
    )

    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=60,
        maxDepth=5,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=int(project_cfg["project"]["seed"])
    )

    models = {
        "log_reg": lr,
        "gbt": gbt
    }

    alert_rate = float(thresh_cfg["operating_points"]["primary"]["alert_rate"])
    extra_rates = thresh_cfg.get("alert_rates", [0.05, 0.10, 0.20])

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results = {
        "run_id": run_id,
        "class_stats": class_stats,
        "feature_count": len(feature_cols),
        "feature_cols_preview": feature_cols[:25],
        "models": {}
    }

    for name, estimator in models.items():
        print(f"\n=== Training: {name} ===")

        pipeline = Pipeline(stages=[assembler, estimator])
        fitted = pipeline.fit(train_w)

        # Predict
        val_pred = fitted.transform(val)
        test_pred = fitted.transform(test)

        # extract probability of class 1 into a scalar for top-k sorting
        val_pred = val_pred.withColumn("prob", vector_to_array("probability")[1])
        test_pred = test_pred.withColumn("prob", vector_to_array("probability")[1])

        val_metrics = evaluate(val_pred, label_col="label")
        test_metrics = evaluate(test_pred, label_col="label")

        val_topk = [topk_metrics(val_pred.select("label", "prob"), k_rate=r, score_col="prob") for r in extra_rates]
        test_topk = [topk_metrics(test_pred.select("label", "prob"), k_rate=r, score_col="prob") for r in extra_rates]

        results["models"][name] = {
            "val": {"metrics": val_metrics, "topk": val_topk},
            "test": {"metrics": test_metrics, "topk": test_topk}
        }

        # Save model
        model_out = abs_path(os.path.join(reports_dir, "models", f"{name}_{run_id}"))
        # overwrite safe for re-runs
        fitted.write().overwrite().save(model_out)
        results["models"][name]["model_path"] = model_out

        print("VAL AUROC:", val_metrics["auroc"], "AUPRC:", val_metrics["auprc"])
        print("TEST AUROC:", test_metrics["auroc"], "AUPRC:", test_metrics["auprc"])

    # Save metrics JSON
    metrics_path = abs_path(os.path.join(reports_dir, "metrics", f"train_metrics_{run_id}.json"))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Training complete")
    print("Metrics saved:", metrics_path)

    spark.stop()


if __name__ == "__main__":
    main()