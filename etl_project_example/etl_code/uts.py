from pyspark.sql.functions import col, lit


def uts_function(df):
    return (
        df
        .filter(col("fare_amount") >= 2.5)
        .select("trip_id", "trip_distance", "fare_amount", "vendor_id")
        .withColumnRenamed("vendor_id", "provider_id")
        .drop("provider_id")
        .orderBy(col("fare_amount").desc())
        .limit(3)
    )