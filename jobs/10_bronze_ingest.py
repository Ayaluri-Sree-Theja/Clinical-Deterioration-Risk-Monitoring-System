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
        .appName(project_cfg["project"]["name"] + "_bronze")
        .config("spark.driver.memory", spark_cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(spark_cfg["shuffle_partitions"]))
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():

    project_cfg = load_yaml("configs/project.yml")

    spark = build_spark(project_cfg)

    raw_dir = project_cfg["paths"]["raw"]
    bronze_dir = project_cfg["paths"]["bronze"]

    os.makedirs(bronze_dir, exist_ok=True)

    def abs_path(p):
        return os.path.abspath(p)

    # Tables to ingest
    tables = ["admissions", "vitals", "labs", "outcomes"]

    for t in tables:

        print(f"\nProcessing RAW → BRONZE : {t}")

        raw_path = abs_path(os.path.join(raw_dir, t))
        bronze_path = abs_path(os.path.join(bronze_dir, t))

        df = spark.read.parquet(raw_path)

        # Standardize column casing
        df = df.select([F.col(c).alias(c.lower()) for c in df.columns])

        # Add ingestion metadata
        df = (
            df
            .withColumn("_ingest_ts", F.current_timestamp())
            .withColumn("_source", F.lit("synthetic_generator"))
        )

        # Basic row count audit
        count = df.count()
        print(f"Rows: {count}")

        (
            df
            .repartition(16)
            .write
            .mode("overwrite")
            .parquet(bronze_path)
        )

    print("\n✅ Bronze ingestion complete")
    spark.stop()


if __name__ == "__main__":
    main()