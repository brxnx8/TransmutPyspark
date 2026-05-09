import os
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .master("local[*]")
    .appName("UTS Source")
    .getOrCreate())

# Importe das libs e funções necessarias
from pyspark.sql.functions import col, lit


def uts_function(df):
    """
    Aplica um pipeline de transformações unárias consecutivas sem
    dependência de coluna entre elas — pares elegíveis para swap pelo UTS.

    Pares presentes (nenhum tem dependência entre si):
      1. filter  → select   (linhas 1-2)
      2. withColumnRenamed → drop  (linhas 3-4)
      3. orderBy → limit          (linhas 5-6)
    """
    return (
        df
        # Par 1 — filter → select  (sem dependência)
        .filter(col("fare_amount") >= 2.5)
        .select("trip_id", "trip_distance", "fare_amount", "vendor_id")
        # Par 2 — withColumnRenamed → drop  (sem dependência)
        .withColumnRenamed("vendor_id", "provider_id")
        .drop("provider_id")
        # Par 3 — orderBy → limit  (sem dependência)
        .orderBy(col("fare_amount").desc())
        .limit(3)
    )