import pytest
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, TimestampType, LongType, StringType,
)    

@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("mutation-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

def _make_valid_trips_df(spark):
    schema = StructType([
        StructField("payment_type",  IntegerType(), True),
        StructField("trip_distance", DoubleType(),  True),
        StructField("tip_amount",    DoubleType(),  True),
        StructField("total_amount",  DoubleType(),  True),
    ])
    data = [
        (1,  2.0, 1.0, 10.0),   # válida  → tip_rate = 0.1
        (2,  3.0, 0.5,  8.0),   # payment_type != 1 → filtrada
        (1,  0.0, 0.0,  5.0),   # trip_distance == 0 → filtrada
        (1, -1.0, 0.0,  5.0),   # trip_distance < 0  → filtrada
        (1,  1.0, 2.0, 20.0),   # válida  → tip_rate = 0.1
    ]
    return spark.createDataFrame(data, schema)

def _make_revenue_df(spark):
    schema = ["VendorID", "total_amount"]
    data = [
        (1, 10.0),
        (1, 20.0),
        (2,  5.0),
        (2, -3.0),   # negativo → filtrado
        (2,  0.0),   # zero → filtrado (> 0, não >=)
    ]
    return spark.createDataFrame(data, schema)

def _make_zone_df(spark):
    schema = StructType([
        StructField("PULocationID",          IntegerType(),   True),
        StructField("passenger_count",       IntegerType(),   True),
        StructField("trip_distance",         DoubleType(),    True),
        StructField("tpep_pickup_datetime",  TimestampType(), True),
        StructField("tpep_dropoff_datetime", TimestampType(), True),
    ])
    t0 = datetime(2023, 1, 1, 8, 0, 0)
    t1 = datetime(2023, 1, 1, 8, 30, 0)   # 30 min depois
    t2 = datetime(2023, 1, 1, 9, 0, 0)    # 60 min depois
    data = [
        # zona 1: 2 corridas válidas de 30 min, 3 milhas → speed = 6 mph
        (1, 1, 3.0, t0, t1),
        (1, 2, 3.0, t0, t1),
        # zona 2: 1 corrida de 60 min, 60 milhas → speed = 60 mph
        (2, 1, 60.0, t0, t2),
        # filtradas:
        (3, 0, 5.0, t0, t1),   # passenger_count = 0
        (3, 1, 0.0, t0, t1),   # trip_distance = 0
        (3, 1, 5.0, t1, t0),   # pickup > dropoff → duration negativa
    ]
    return spark.createDataFrame(data, schema)

def _make_tip_ranking_df(spark):
    schema = StructType([
        StructField("VendorID",              IntegerType(),   True),
        StructField("tip_amount",            DoubleType(),    True),
        StructField("tpep_pickup_datetime",  TimestampType(), True),
    ])
    h8  = datetime(2023, 1, 1, 8,  0, 0)
    h10 = datetime(2023, 1, 1, 10, 0, 0)
    data = [
        # hora 8: 5 corridas com gorjeta — top-3 são 9, 7, 5
        (1,  9.0, h8),
        (1,  7.0, h8),
        (1,  5.0, h8),
        (1,  3.0, h8),
        (1,  1.0, h8),
        # hora 8: gorjeta zero → filtrada
        (1,  0.0, h8),
        # hora 10: apenas 2 corridas → ambas aparecem
        (2,  4.0, h10),
        (2,  2.0, h10),
    ]
    return spark.createDataFrame(data, schema)

def _make_efficiency_df(spark):
    schema = StructType([
        StructField("PULocationID",          IntegerType(),   True),
        StructField("fare_amount",           DoubleType(),    True),
        StructField("trip_distance",         DoubleType(),    True),
        StructField("passenger_count",       IntegerType(),   True),
        StructField("tpep_pickup_datetime",  TimestampType(), True),
        StructField("tpep_dropoff_datetime", TimestampType(), True),
    ])
    # pickup às 9h (morning), 30 min de duração → speed=120 mph → filtrado por speed<150
    t_morn_s = datetime(2023, 1, 1,  9,  0, 0)
    t_morn_e = datetime(2023, 1, 1,  9, 30, 0)
    # pickup às 15h (afternoon), 60 min → speed=30 mph, cost_per_mile=5.0
    t_aft_s  = datetime(2023, 1, 1, 15,  0, 0)
    t_aft_e  = datetime(2023, 1, 1, 16,  0, 0)
    # pickup à meia-noite (overnight)
    t_ngt_s  = datetime(2023, 1, 1,  0,  0, 0)
    t_ngt_e  = datetime(2023, 1, 1,  0, 30, 0)

    data = [
        # zona 1, afternoon: 30 milhas em 60 min = 30 mph, fare=150 → cost=5.0/mi
        (1, 150.0, 30.0, 1, t_aft_s, t_aft_e),
        (1, 150.0, 30.0, 2, t_aft_s, t_aft_e),
        (1, 150.0, 30.0, 1, t_aft_s, t_aft_e),
        # zona 1, morning: 60 milhas em 30 min = 120 mph, fare=300 → cost=5.0/mi
        (1, 300.0, 60.0, 1, t_morn_s, t_morn_e),
        # zona 2, overnight: 10 milhas em 30 min = 20 mph, fare=50 → cost=5.0/mi
        (2,  50.0, 10.0, 1, t_ngt_s, t_ngt_e),
        # filtradas pelo sanity filter inicial:
        (3,  -5.0, 10.0, 1, t_aft_s, t_aft_e),   # fare negativo
        (3,   5.0,  0.0, 1, t_aft_s, t_aft_e),   # distance zero
        (3,   5.0, 10.0, 0, t_aft_s, t_aft_e),   # passenger zero
    ]
    return spark.createDataFrame(data, schema)