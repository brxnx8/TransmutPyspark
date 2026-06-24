import os
import sys
import ctypes
import pytest
import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.testing.utils import assertDataFrameEqual
from nftp import nftp_function


# ── Fixture Spark ─────────────────────────────────────────────────────────────

def _short_path(path):
    """Converte caminho para formato 8.3 (evita espaços e acentos)."""
    buf = ctypes.create_unicode_buffer(260)
    ret = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 260)
    if ret == 0:
        return path
    return buf.value


@pytest.fixture(scope="session")
def spark():
    # 1. SPARK_HOME usando o pacote pyspark (caminho curto)
    pyspark_home = os.path.dirname(pyspark.__file__)
    os.environ["SPARK_HOME"] = _short_path(pyspark_home)

    # 2. Evitar problema com underscore no hostname
    os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

    # 3. 🎯 Força o Spark a usar o Python do ambiente virtual da sandbox
    #    (resolve "Python não foi encontrado" e timeouts do worker)
    os.environ["PYSPARK_PYTHON"] = sys.executable

    spark_session = (
        SparkSession.builder
        .master("local[1]")
        .appName("pytest-pyspark-mtr-weak")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
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