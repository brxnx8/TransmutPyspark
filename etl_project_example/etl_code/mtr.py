from pyspark.sql.functions import *
from pyspark.sql.window import Window

def mtr_function(df):
    return (
        df.withColumn(
            "trip_duration_min",
            (unix_timestamp(col("tpep_dropoff_datetime")) -
             unix_timestamp(col("tpep_pickup_datetime"))) / 60
        )
        .withColumn(
            "valor_calculado",
            coalesce(col("fare_amount"), lit(0.0)) +
            coalesce(col("extra"), lit(0.0)) +
            coalesce(col("mta_tax"), lit(0.0)) +
            coalesce(col("tip_amount"), lit(0.0)) +
            coalesce(col("tolls_amount"), lit(0.0)) +
            coalesce(col("improvement_surcharge"), lit(0.0)) +
            coalesce(col("congestion_surcharge"), lit(0.0)) +
            coalesce(col("airport_fee"), lit(0.0))
        )
        .withColumn(
            "valor_por_km",
            when(
                col("trip_distance") > 0,
                col("fare_amount") / col("trip_distance")
            ).otherwise(lit(None))
        )
    )