# etl_project_example/etl_test/conftest.py
import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("Transmut-Test-Suite") \
        .getOrCreate()
    yield spark
    spark.stop()