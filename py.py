from pyspark.sql import SparkSession
import os, shutil

spark = (SparkSession.builder
         .master("local[2]")
         .appName("write_test")
         .config("spark.driver.memory","2g")
         .config("spark.sql.shuffle.partitions","16")
         .getOrCreate())

out = "data/_tmp_parquet"
if os.path.exists(out):
    shutil.rmtree(out)

spark.range(1000).write.mode("overwrite").parquet(out)
print("rows:", spark.read.parquet(out).count())
spark.stop()
