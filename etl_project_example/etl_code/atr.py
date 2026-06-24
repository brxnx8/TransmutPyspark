from pyspark.sql.functions import col, sum, count, avg, max, min, rank, dense_rank
from pyspark.sql.window import Window


def atr_function(df):

    agg_df = (df
        .groupBy("driver_id", "region")
        .agg(
            sum("fare_amount").alias("total_fare"),

            avg("tip_amount").alias("avg_tip"),

            max("trip_distance").alias("max_distance"),
        )
    )

    trip_count_df = (df
        .groupBy("driver_id", "region")
        .count()
        .withColumnRenamed("count", "trip_count")
    )

    window_spec = Window.partitionBy("region").orderBy(col("total_fare").desc())
    ranked_df = agg_df.withColumn("fare_rank", rank().over(window_spec))

    result = ranked_df.join(
        trip_count_df.select("driver_id", "region", "trip_count"),
        on=["driver_id", "region"],
        how="inner"
    )

    return result.select(
        "driver_id", "region",
        "total_fare", "avg_tip", "max_distance",
        "trip_count", "fare_rank"
    )