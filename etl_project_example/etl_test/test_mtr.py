import os
import sys
import ctypes
import pytest
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from mtr import mtr_function

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


# ── Schema de entrada ─────────────────────────────────────────────────────────

def _input_schema():
    return StructType([
        StructField("tpep_pickup_datetime",  StringType(), True),
        StructField("tpep_dropoff_datetime", StringType(), True),
        StructField("fare_amount",           DoubleType(), True),
        StructField("extra",                 DoubleType(), True),
        StructField("mta_tax",               DoubleType(), True),
        StructField("tip_amount",            DoubleType(), True),
        StructField("tolls_amount",          DoubleType(), True),
        StructField("improvement_surcharge", DoubleType(), True),
        StructField("congestion_surcharge",  DoubleType(), True),
        StructField("airport_fee",           DoubleType(), True),
        StructField("trip_distance",         DoubleType(), True),
    ])


# ── Teste 1: apenas verifica que as colunas novas existem ────────────────────
# MUTANTES VIVOS:
#   MTR — todas as expressões de withColumn podem ser trocadas por 0, 1, -1, None
#   sem que o teste falhe, pois só checa presença das colunas, não os valores.

def test_colunas_novas_existem(spark):
    data = [
        ("2023-01-01 10:00:00", "2023-01-01 10:30:00",
         10.0, 1.0, 0.5, 2.0, 0.0, 0.3, 2.5, 0.0, 5.0),
    ]
    df = spark.createDataFrame(data, _input_schema())
    result = mtr_function(df)
    cols = result.columns
    assert "trip_duration_min"  in cols
    assert "valor_calculado"    in cols
    assert "valor_por_km"       in cols


# ── Teste 2: verifica trip_duration_min apenas com asserção fraca (> 0) ──────
# MUTANTES VIVOS:
#   MTR — substitui a expressão inteira por lit(1): 1 > 0 → passa.
#   MTR — substitui por negated (resultado negativo p/ viagem normal): falha aqui,
#         mas não falha p/ viagens de duração zero (covered abaixo).
#   MTR — substitui por identity col("tpep_dropoff_datetime"): não numérico,
#         mas o teste só faz assertIsNotNone, não checa tipo.

def test_trip_duration_positivo(spark):
    data = [
        ("2023-01-01 10:00:00", "2023-01-01 10:30:00",
         10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
    ]
    df = spark.createDataFrame(data, _input_schema())
    result = mtr_function(df)
    row = result.collect()[0]
    # Só checa se é positivo — não valida o valor exato (30.0)
    # Mutante lit(1) → 1 > 0 → sobrevive
    # Mutante lit(60) → 60 > 0 → sobrevive
    assert row["trip_duration_min"] > 0


# ── Teste 3: verifica valor_calculado só com uma soma parcial ─────────────────
# MUTANTES VIVOS:
#   MTR — remove qualquer parcela individual do somatório (ex: coalesce(extra, 0)
#         substituído por lit(0)): como o dado de entrada já tem extra=0.0,
#         o resultado não muda e o mutante sobrevive.
#   MTR — substitui coalesce(airport_fee, 0) por lit(0): airport_fee=0 no dado,
#         então o valor calculado não se altera.
#   MTR — substitui coalesce(tolls_amount, 0) por lit(0): mesmo caso.

def test_valor_calculado_basico(spark):
    data = [
        # extra, mta_tax, tip, tolls, improvement, congestion, airport_fee todos = 0
        # Assim valor_calculado == fare_amount == 10.0
        # Qualquer mutante que zere UMA parcela que já era zero sobrevive.
        ("2023-01-01 10:00:00", "2023-01-01 10:30:00",
         10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
    ]
    df = spark.createDataFrame(data, _input_schema())
    result = mtr_function(df)
    row = result.collect()[0]
    assert row["valor_calculado"] == 10.0


# ── Teste 4: não testa valor_por_km de forma alguma ──────────────────────────
# MUTANTES VIVOS:
#   MTR — substitui col("fare_amount") / col("trip_distance") por lit(0) → sobrevive.
#   MTR — substitui por lit(1) → sobrevive.
#   MTR — substitui por lit(None) → sobrevive.
#   MTR — substitui por negated → sobrevive.
#   MTR — substitui por identity col("fare_amount") → sobrevive.
# Nenhuma asserção sobre o valor de valor_por_km é feita.

def test_schema_resultado(spark):
    data = [
        ("2023-01-01 10:00:00", "2023-01-01 10:30:00",
         12.0, 1.0, 0.5, 2.0, 0.0, 0.3, 2.5, 0.0, 4.0),
    ]
    df = spark.createDataFrame(data, _input_schema())
    result = mtr_function(df)
    # Só valida que o DataFrame tem mais colunas que o original
    assert len(result.columns) == len(df.columns) + 3


# ── Teste 5: caso com None em fare_amount, mas sem checar valor_por_km ────────
# MUTANTES VIVOS:
#   MTR — substitui coalesce(col("fare_amount"), lit(0.0)) por col("fare_amount"):
#         quando fare_amount=None, identity retorna None, mas o teste não
#         checa o valor de valor_calculado neste caso — apenas que não lança erro.
#   MTR — substitui a expressão inteira de valor_por_km por lit(0): sobrevive
#         porque não há asserção sobre esse campo.

def test_sem_erro_com_nulos(spark):
    data = [
        ("2023-01-01 13:00:00", "2023-01-01 13:10:00",
         None, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0),
    ]
    df = spark.createDataFrame(data, _input_schema())
    # Apenas garante que a função não lança exceção
    result = mtr_function(df)
    assert result.count() == 1