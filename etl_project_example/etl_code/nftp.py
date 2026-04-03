import os
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').appName("Iniciando com Spark").getOrCreate()
"#Importe das libs e funções necessarias"
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def nftp_function(df):
    return df.filter(col("trip_distance").isNotNull() & col("fare_amount").isNotNull() & (col("trip_distance") > 0.0) & (col("fare_amount") >= 2.5))