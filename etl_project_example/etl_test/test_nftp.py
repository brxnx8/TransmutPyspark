import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.testing.utils import assertDataFrameEqual
from etl_project_example.etl_code.etl import atr_fuction


@pytest.fixture(scope="session")
def spark():
    # Inicialização limpa e moderna
    spark_session = (SparkSession.builder
        .master("local[1]")
        .appName("pytest-pyspark-local")
        .config("spark.ui.enabled", "false")
        .getOrCreate())

    yield spark_session
    spark_session.stop()



def test_nftp_function_filtros_e_limites(spark):
    # 1. Preparação (Arrange)
    # Usaremos uma coluna "trip_id" apenas para facilitar o rastreio visual de quais linhas sobreviveram
    schema = StructType([
        StructField("trip_id", StringType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("fare_amount", DoubleType(), True)
    ])

    input_data = [
        ("1", 5.0, 15.0),

        ("2", 1.0, 2.50),

        ("3", 0.0, 10.0),

        ("4", -1.5, 10.0),

        ("5", 2.0, 2.49),

        ("6", None, 5.0),

        ("7", 3.0, None),

        ("8", None, None)
    ]

    input_df = spark.createDataFrame(input_data, schema)

    expected_data = [
        ("1", 5.0, 15.0),
        ("2", 1.0, 2.50)
    ]

    expected_df = spark.createDataFrame(expected_data, schema)

    result_df = nftp_function(input_df)

    assertDataFrameEqual(result_df, expected_df)