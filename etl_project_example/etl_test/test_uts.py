import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType
)
from pyspark.testing.utils import assertDataFrameEqual

from uts import uts_function


@pytest.fixture(scope="session")
def spark():
    spark_session = (SparkSession.builder
        .master("local[1]")
        .appName("pytest-pyspark-uts")
        .config("spark.ui.enabled", "false")
        .getOrCreate())
    yield spark_session
    spark_session.stop()


# ---------------------------------------------------------------------------
# Schema e dados de entrada compartilhados
# ---------------------------------------------------------------------------
SCHEMA = StructType([
    StructField("trip_id",       StringType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("fare_amount",   DoubleType(), True),
    StructField("vendor_id",     StringType(), True),
])

INPUT_DATA = [
    ("1",   5.0,   15.0,  "A"),
    ("2",   1.0,    2.50, "B"),
    ("3",   0.0,   10.0,  "A"),
    ("4",  -1.5,   12.0,  "C"),
    ("5",   2.0,    2.49, "A"),
    ("6",   3.0,    8.0,  "B"),
    ("7",   4.0,   20.0,  "C"),
    ("8",   6.0,    5.0,  "A"),
]

SCHEMA_EXPECTED = StructType([
    StructField("trip_id",       StringType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("fare_amount",   DoubleType(), True),
])

EXPECTED_DATA = [
    ("7", 4.0, 20.0),
    ("1", 5.0, 15.0),
    ("4", -1.5, 12.0),
]


def test_uts_pipeline_result(spark):
    input_df    = spark.createDataFrame(INPUT_DATA, SCHEMA)
    expected_df = spark.createDataFrame(EXPECTED_DATA, SCHEMA_EXPECTED)

    result_df = uts_function(input_df)

    assertDataFrameEqual(result_df, expected_df)


def test_uts_filter_before_select_matters(spark):

    input_df = spark.createDataFrame(INPUT_DATA, SCHEMA)
    result_df = uts_function(input_df)

    assert result_df.columns == ["trip_id", "trip_distance", "fare_amount"], (
        "Pipeline deve entregar apenas trip_id, trip_distance e fare_amount"
    )

def test_uts_filtered_row_absent(spark):
    input_df  = spark.createDataFrame(INPUT_DATA, SCHEMA)
    result_df = uts_function(input_df)

    trip_ids = [row.trip_id for row in result_df.collect()]
    assert "5" not in trip_ids, (
        "trip_id='5' (fare_amount=2.49) deve ter sido removida pelo filter"
    )

def test_uts_limit_respects_order(spark):
    input_df  = spark.createDataFrame(INPUT_DATA, SCHEMA)
    result_df = uts_function(input_df)

    fares = [row.fare_amount for row in result_df.collect()]

    assert sorted(fares, reverse=True) == [20.0, 15.0, 12.0], (
        "limit(3) deve pegar os 3 maiores fare_amounts (orderBy aplicado antes)"
    )


def test_uts_row_count(spark):
    input_df  = spark.createDataFrame(INPUT_DATA, SCHEMA)
    result_df = uts_function(input_df)

    assert result_df.count() == 3, "Pipeline deve retornar exatamente 3 linhas"