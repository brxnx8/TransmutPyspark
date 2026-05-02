import os
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .master("local[*]")
    .appName("ATR Example")
    .getOrCreate())

from pyspark.sql.functions import col, sum, count, avg, max, min, rank, dense_rank
from pyspark.sql.window import Window


def atr_function(df):
    """
    Recebe um DataFrame com colunas:
        driver_id   (string)  – identificador do motorista
        region      (string)  – região da corrida
        trip_distance (double) – distância percorrida
        fare_amount   (double) – valor da tarifa
        tip_amount    (double) – gorjeta

    Retorna um DataFrame com:
        driver_id, region,
        total_fare      – soma das tarifas por motorista+região          [Caso A1+A2]
        trip_count      – contagem de corridas por motorista+região      [Caso B shorthand]
        avg_tip         – média de gorjetas por motorista+região         [Caso A1]
        max_distance    – maior distância por motorista+região           [Caso A1]
        fare_rank       – ranking da tarifa total dentro da região       [Caso C]
    """

    # ── Caso A: .groupBy().agg() com função de agregação explícita ────────
    # A1 – função de agregação trocável (sum, avg, max)
    # A2 – coluna de entrada trocável ("fare_amount", "tip_amount", "trip_distance")
    agg_df = (df
        .groupBy("driver_id", "region")            # Caso D – 2 chaves removíveis
        .agg(
            sum("fare_amount").alias("total_fare"),  # A1: sum→{count,avg,max,min,...}
                                                     # A2: "fare_amount"→{"tip_amount","trip_distance",...}
            avg("tip_amount").alias("avg_tip"),      # A1: avg→{sum,count,max,min,...}
                                                     # A2: "tip_amount"→{"fare_amount","trip_distance",...}
            max("trip_distance").alias("max_distance"),  # A1: max→{sum,count,avg,min,...}
        )
    )

    # ── Caso B: shorthand .groupBy().count() ─────────────────────────────
    # B – count trocável por sum, avg, max, min, first, last, mean
    trip_count_df = (df
        .groupBy("driver_id", "region")             # Caso D – 2 chaves removíveis
        .count()                                     # B: count→{sum,avg,max,...}
        .withColumnRenamed("count", "trip_count")
    )

    # ── Caso C: função de janela ─────────────────────────────────────────
    # C – rank trocável por dense_rank, row_number, percent_rank, cume_dist
    window_spec = Window.partitionBy("region").orderBy(col("total_fare").desc())
    ranked_df = agg_df.withColumn("fare_rank", rank().over(window_spec))  # C: rank→{dense_rank,...}

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