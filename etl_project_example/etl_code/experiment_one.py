from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

def valid_trips_with_tip_rate(df: DataFrame) -> DataFrame:
    """
    Mantém apenas corridas pagas em dinheiro com distância positiva e
    adiciona a coluna tip_rate (gorjeta / valor total da corrida).

    Alvos por operador
    ------------------
    MTR  : withColumn("tip_rate", ...)
    NFTP : filter com dois predicados (payment_type == 1, trip_distance > 0.0)
    ATR  : —  (sem agregação; a coluna nova usa divisão simples)
    UTS  : filter → withColumn  (sem dependência; swap elegível)
    """
    df = df.filter(
        (F.col("payment_type") == 1) & (F.col("trip_distance") > 0.0)
    )
    df = df.withColumn(
        "tip_rate",
        F.col("tip_amount") / F.col("total_amount"),
    )
    return df

def revenue_by_vendor(df: DataFrame) -> DataFrame:
    """
    Filtra corridas com valor positivo e agrega receita e contagem
    de corridas por VendorID.

    Alvos por operador
    ------------------
    MTR  : select final
    NFTP : filter (total_amount > 0.0)
    ATR  : groupBy("VendorID").agg(sum + count)
    UTS  : filter → select  (sem dependência; swap elegível)
    """
    df = df.filter(F.col("total_amount") > 0.0)
    df = df.groupBy("VendorID").agg(
        F.sum("total_amount").alias("total_revenue"),
        F.count("VendorID").alias("trip_count"),
    )
    df = df.select("VendorID", "total_revenue", "trip_count")
    return df

def trip_profile_by_zone(df: DataFrame) -> DataFrame:
    """
    Para cada zona de embarque (PULocationID), calcula métricas de duração,
    distância e passageiros, excluindo corridas suspeitas.

    Alvos por operador
    ------------------
    MTR  : withColumn("trip_duration_min", ...) e withColumn("speed_mph", ...)
    NFTP : filter com três condições (passenger_count, trip_distance, duration)
    ATR  : groupBy("PULocationID").agg(avg + max + count)
    UTS  : filter → withColumn("trip_duration_min") → withColumn("speed_mph")
           (pares sem dependência são elegíveis para swap)
    """
    df = df.withColumn(
        "trip_duration_min",
        (F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long")) / 60.0,
    )
    df = df.filter(
        (F.col("passenger_count") > 0)
        & (F.col("trip_distance") > 0.0)
        & (F.col("trip_duration_min") > 0.0)
    )
    df = df.withColumn(
        "speed_mph",
        F.col("trip_distance") / (F.col("trip_duration_min") / 60.0),
    )
    df = df.filter(F.col("speed_mph") < 100.0)
    df = df.groupBy("PULocationID").agg(
        F.avg("trip_duration_min").alias("avg_duration_min"),
        F.avg("trip_distance").alias("avg_distance_miles"),
        F.max("speed_mph").alias("max_speed_mph"),
        F.count("PULocationID").alias("trip_count"),
    )
    return df

def tip_ranking_by_hour(df: DataFrame) -> DataFrame:
    """
    Adiciona a hora do dia, descarta gorjetas ausentes ou nulas, e rankeia
    cada corrida dentro da sua hora pelo valor de gorjeta (maior = rank 1).
    Retorna apenas o top-3 por hora.

    Alvos por operador
    ------------------
    MTR  : withColumn("pickup_hour", ...) e withColumn("tip_rank", ...)
    NFTP : filter (tip_amount > 0.0) e where (tip_rank <= 3)
    ATR  : rank().over(window) — função de janela trocável por dense_rank etc.
    UTS  : withColumn("pickup_hour") → filter  (sem dependência; swap elegível)
    """
    df = df.withColumn(
        "pickup_hour",
        F.hour(F.col("tpep_pickup_datetime")),
    )
    df = df.filter(F.col("tip_amount") > 0.0)

    window_spec = Window.partitionBy("pickup_hour").orderBy(F.col("tip_amount").desc())

    df = df.withColumn("tip_rank", F.rank().over(window_spec))
    df = df.where(F.col("tip_rank") <= 3)
    df = df.select("pickup_hour", "tip_amount", "tip_rank", "VendorID")
    return df