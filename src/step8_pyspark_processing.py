import time
import pandas as pd

print("\n===== STEP 8 : PANDAS VS PYSPARK =====\n")

# ----------------------------
# PANDAS TIMING
# ----------------------------
start_pandas = time.time()

df_pandas = pd.read_csv(
    "data/ecommerce.csv",
    nrows=500000
)

df_pandas["purchase"] = df_pandas["event_type"].apply(
    lambda x: 1 if str(x).lower() == "purchase" else 0
)

pandas_rows = len(df_pandas)

end_pandas = time.time()

pandas_time = round(end_pandas - start_pandas, 2)

print(f"Pandas Rows      : {pandas_rows}")
print(f"Pandas Time (s)  : {pandas_time}")

# ----------------------------
# PYSPARK TIMING
# ----------------------------
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
import os

# Force Spark to use local filesystem instead of HDFS
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

start_spark = time.time()

spark = SparkSession.builder \
    .appName("EcommerceBigData") \
    .master("local[*]") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

csv_path = "file://" + os.path.abspath("data/ecommerce.csv")

df_spark = spark.read.csv(
    csv_path,
    header=True,
    inferSchema=True
)

df_spark = df_spark.withColumn(
    "purchase",
    when(col("event_type") == "purchase", 1).otherwise(0)
)

spark_rows = df_spark.count()

end_spark = time.time()

spark_time = round(end_spark - start_spark, 2)

print(f"PySpark Rows     : {spark_rows}")
print(f"PySpark Time (s) : {spark_time}")

# ----------------------------
# SAVE COMPARISON
# ----------------------------
comparison = pd.DataFrame({
    "Framework": ["Pandas", "PySpark"],
    "Rows Processed": [pandas_rows, spark_rows],
    "Time Seconds": [pandas_time, spark_time]
})

comparison.to_csv(
    "outputs/comparison/pyspark_vs_pandas.csv",
    index=False
)

print("\nSaved -> outputs/comparison/pyspark_vs_pandas.csv")

spark.stop()