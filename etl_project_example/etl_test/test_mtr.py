
import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.testing.utils import assertDataFrameEqual
from pyspark.sql.functions import unix_timestamp, col, coalesce, lit, when
from etl_project_example.etl_code.etl import mtr_function


def test_mtr_function_calculos_e_anomalias(spark_session):
    schema = StructType([
        StructField("tpep_pickup_datetime", StringType(), True),
        StructField("tpep_dropoff_datetime", StringType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("extra", DoubleType(), True),
        StructField("mta_tax", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("tolls_amount", DoubleType(), True),
        StructField("improvement_surcharge", DoubleType(), True),
        StructField("congestion_surcharge", DoubleType(), True),
        StructField("airport_fee", DoubleType(), True),
        StructField("trip_distance", DoubleType(), True)
    ])

    input_data = [

        ("2023-01-01 10:00:00", "2023-01-01 10:30:00", 10.0, 1.0, 0.5, 2.0, 0.0, 0.3, 2.5, 0.0, 5.0),

        ("2023-01-01 11:00:00", "2023-01-01 11:15:00", 5.0, None, None, None, None, None, None, None, 0.0),

        (None, None, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),

        ("2023-01-01 12:30:00", "2023-01-01 12:00:00", 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),

        ("2023-01-01 13:00:00", "2023-01-01 13:10:00", None, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0),

        (None, None, None, None, None, None, None, None, None, None, None)
    ]

    input_df = spark_session.createDataFrame(input_data, schema)

    expected_schema = schema.add(StructField("trip_duration_min", DoubleType(), True)) \
                            .add(StructField("valor_calculado", DoubleType(), True)) \
                            .add(StructField("valor_por_km", DoubleType(), True))

    expected_data = [

        ("2023-01-01 10:00:00", "2023-01-01 10:30:00", 10.0, 1.0, 0.5, 2.0, 0.0, 0.3, 2.5, 0.0, 5.0,
         30.0, 16.3, 2.0),

        ("2023-01-01 11:00:00", "2023-01-01 11:15:00", 5.0, None, None, None, None, None, None, None, 0.0,
         15.0, 5.0, None),

        (None, None, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
         None, 10.0, None),

        ("2023-01-01 12:30:00", "2023-01-01 12:00:00", 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,
         -30.0, 15.0, 5.0),

        ("2023-01-01 13:00:00", "2023-01-01 13:10:00", None, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,
         10.0, 1.5, None),

        (None, None, None, None, None, None, None, None, None, None, None,
         None, 0.0, None)
    ]

    expected_df = spark_session.createDataFrame(expected_data, expected_schema)

    result_df = mtr_function(input_df)

    assertDataFrameEqual(result_df, expected_df)
