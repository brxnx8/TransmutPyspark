from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

def full_trip_efficiency_analysis(df: DataFrame) -> DataFrame:
    """
    Pipeline completo que:
      1. Descarta corridas inválidas (valores negativos, duração zero).
      2. Deriva métricas de duração, velocidade e custo por milha.
      3. Detecta corridas anômalas por zona usando desvio do percentil 95.
      4. Rankeia zonas por eficiência de receita.
      5. Agrega o resultado final por zona e turno do dia.

    Alvos por operador
    ------------------
    MTR  : withColumn("trip_duration_min"), withColumn("speed_mph"),
           withColumn("cost_per_mile"), withColumn("shift"), withColumn("is_anomaly"),
           select final
    NFTP : filter inicial (fare_amount, trip_distance, passenger_count),
           filter de velocidade (speed_mph > 0.0 & speed_mph < 150.0),
           where para remover anomalias (is_anomaly == 0)
    ATR  : groupBy("PULocationID").agg(percentile_approx, avg, count)  [passo 3]
           groupBy("PULocationID", "shift").agg(sum, avg, count)  [passo 5]
           percent_rank().over(window) para ranking de zonas
    UTS  : filter → withColumn("trip_duration_min") → withColumn("speed_mph")
           → withColumn("cost_per_mile")  — múltiplos pares sem dependência
           withColumn("shift") → filter de velocidade  (sem dependência)
    """
    
    df = df.filter(
        (F.col("fare_amount") > 0.0)
        & (F.col("trip_distance") > 0.0)
        & (F.col("passenger_count") > 0)
    )

    df = df.withColumn(
        "trip_duration_min",
        (F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long")) / 60.0,
    )
    df = df.filter(
        (F.col("speed_mph") > 0.0) & (F.col("trip_duration_min") > 0.0)
    )
    df = df.withColumn(
        "speed_mph",
        F.col("trip_distance") / (F.col("trip_duration_min") / 60.0),
    )
    df = df.filter(F.col("speed_mph") < 150.0)
    df = df.withColumn(
        "cost_per_mile",
        F.col("fare_amount") / F.col("trip_distance"),
    )

    df = df.withColumn(
        "shift",
        F.when(F.hour(F.col("tpep_pickup_datetime")) < 6, "overnight")
         .when(F.hour(F.col("tpep_pickup_datetime")) < 12, "morning")
         .when(F.hour(F.col("tpep_pickup_datetime")) < 18, "afternoon")
         .otherwise("night"),
    )

    zone_stats = df.groupBy("PULocationID").agg(
        F.avg("cost_per_mile").alias("avg_cost_per_mile"),
        F.count("PULocationID").alias("zone_trip_count"),
    )

    window_pct = Window.partitionBy("PULocationID")
    df = df.withColumn(
        "p95_cost",
        F.percentile_approx(F.col("cost_per_mile"), 0.95).over(window_pct),
    )
    df = df.withColumn(
        "is_anomaly",
        F.when(F.col("cost_per_mile") > F.col("p95_cost"), 1).otherwise(0),
    )
    df = df.where(F.col("is_anomaly") == 0)

    window_rank = Window.orderBy(F.col("avg_cost_per_mile").desc())
    df = df.join(zone_stats, on="PULocationID", how="left")
    df = df.withColumn(
        "zone_revenue_rank",
        F.percent_rank().over(window_rank),
    )

    df = df.groupBy("PULocationID", "shift").agg(
        F.sum("fare_amount").alias("total_fare"),
        F.avg("trip_distance").alias("avg_distance"),
        F.avg("cost_per_mile").alias("avg_cost_per_mile"),
        F.count("PULocationID").alias("trip_count"),
    )

    df = df.select(
        "PULocationID",
        "shift",
        "total_fare",
        "avg_distance",
        "avg_cost_per_mile",
        "trip_count",
    )
    return df