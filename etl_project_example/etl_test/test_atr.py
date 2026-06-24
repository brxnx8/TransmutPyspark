import os
import sys
import ctypes
import pytest
import pyspark
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, IntegerType
)
from pyspark.testing.utils import assertDataFrameEqual

from atr import atr_function


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

INPUT_SCHEMA = StructType([
    StructField("driver_id",      StringType(), True),
    StructField("region",         StringType(), True),
    StructField("trip_distance",  DoubleType(), True),
    StructField("fare_amount",    DoubleType(), True),
    StructField("tip_amount",     DoubleType(), True),
])

OUTPUT_SCHEMA = StructType([
    StructField("driver_id",     StringType(), True),
    StructField("region",        StringType(), True),
    StructField("total_fare",    DoubleType(), True),
    StructField("avg_tip",       DoubleType(), True),
    StructField("max_distance",  DoubleType(), True),
    StructField("trip_count",    LongType(),   True),
    StructField("fare_rank",     IntegerType(), True),
])


# ── Dados de entrada ──────────────────────────────────────────────────────────
INPUT_DATA = [
    ("D1", "Norte",  5.0,  10.0,  1.0),
    ("D1", "Norte", 10.0,  20.0,  2.0),
    ("D1", "Norte", 15.0,  30.0,  3.0),
    ("D2", "Norte",  2.0,  10.0,  4.0),
    ("D2", "Norte",  4.0,  20.0,  5.0),
    ("D2", "Norte",  6.0,  30.0,  6.0),
    ("D3", "Sul",   20.0,  50.0,  8.0),
    ("D3", "Sul",   25.0,  70.0, 10.0),
]

# ── Expected ──────────────────────────────────────────────────────────────────
EXPECTED_DATA = [
    # driver_id, region, total_fare, avg_tip, max_distance, trip_count, fare_rank
    ("D1", "Norte", 60.0,  2.0, 15.0, 3, 1),
    ("D2", "Norte", 60.0,  5.0,  6.0, 3, 1),
    ("D3", "Sul",  120.0,  9.0, 25.0, 2, 1),
    ("D4", "Norte", 30.0,  7.0,  3.0, 1, 3),
]

_D4_INPUT = [
    ("D4", "Norte", 3.0, 30.0, 7.0),
]

# ── Teste principal ───────────────────────────────────────────────────────────

def test_atr_function_aggregations(spark):
    all_input = INPUT_DATA + _D4_INPUT
    input_df = spark.createDataFrame(all_input, INPUT_SCHEMA)

    result_df = atr_function(input_df)

    expected_df = spark.createDataFrame(EXPECTED_DATA, OUTPUT_SCHEMA)

    assertDataFrameEqual(result_df, expected_df)

def test_atr_groupby_keys_both_required(spark):
    all_input = INPUT_DATA + _D4_INPUT
    input_df = spark.createDataFrame(all_input, INPUT_SCHEMA)

    result_df = atr_function(input_df).cache()

    assert result_df.count() == 4, (
        "Esperado 4 grupos distintos (D1/Norte, D2/Norte, D3/Sul, D4/Norte). "
        "Se uma chave do groupBy for removida os grupos se fundem."
    )

    norte_rows = result_df.filter(col("region") == "Norte").orderBy("driver_id").collect()
    assert len(norte_rows) == 3, (
        "Esperados 3 grupos em 'Norte' (D1, D2, D4). "
        "Remoção de driver_id do groupBy funde esses grupos."
    )

    d1 = next(r for r in norte_rows if r["driver_id"] == "D1")
    assert d1["total_fare"] == 60.0, f"D1 total_fare esperado 60.0, obtido {d1['total_fare']}"
    assert d1["trip_count"] == 3,    f"D1 trip_count esperado 3, obtido {d1['trip_count']}"

    d2 = next(r for r in norte_rows if r["driver_id"] == "D2")
    assert d2["total_fare"] == 60.0, f"D2 total_fare esperado 60.0, obtido {d2['total_fare']}"
    assert d2["trip_count"] == 3,    f"D2 trip_count esperado 3, obtido {d2['trip_count']}"

    sul_rows = result_df.filter(col("region") == "Sul").collect()
    assert len(sul_rows) == 1, (
        "Esperado exatamente 1 grupo em 'Sul' (D3). "
        "Remoção de region do groupBy pode fundir regiões."
    )
    d3 = sul_rows[0]
    assert d3["total_fare"] == 120.0, f"D3 total_fare esperado 120.0, obtido {d3['total_fare']}"


def test_atr_window_rank_vs_dense_rank(spark):

    all_input = INPUT_DATA + _D4_INPUT
    input_df = spark.createDataFrame(all_input, INPUT_SCHEMA)

    result_df = atr_function(input_df)

    norte = result_df.filter(col("region") == "Norte").orderBy("driver_id").collect()
    d1 = next(r for r in norte if r["driver_id"] == "D1")
    d2 = next(r for r in norte if r["driver_id"] == "D2")
    d4 = next(r for r in norte if r["driver_id"] == "D4")

    assert d1["fare_rank"] == 1, (
        f"D1 fare_rank esperado 1 (empatado com D2), obtido {d1['fare_rank']}. "
        "Indica que a função de janela foi trocada."
    )
    assert d2["fare_rank"] == 1, (
        f"D2 fare_rank esperado 1 (empatado com D1), obtido {d2['fare_rank']}. "
        "Indica que a função de janela foi trocada."
    )

    assert d4["fare_rank"] == 3, (
        f"D4 fare_rank esperado 3 com rank(), obtido {d4['fare_rank']}. "
        "Se o valor for 2, a função foi mutada para dense_rank(). "
        "Se D1 e D2 não tiverem o mesmo rank, pode ser row_number()."
    )


def test_atr_aggregation_values_are_distinct(spark):
    all_input = INPUT_DATA + _D4_INPUT
    input_df = spark.createDataFrame(all_input, INPUT_SCHEMA)

    result_df = atr_function(input_df)

    for row in result_df.collect():
        vals = {
            "total_fare":   row["total_fare"],
            "avg_tip":      row["avg_tip"],
            "max_distance": row["max_distance"],
        }
        assert len(set(vals.values())) == len(vals), (
            f"Valores de agregação não são todos distintos para "
            f"driver={row['driver_id']} region={row['region']}: {vals}. "
            "Isso indica que duas funções de agregação produzem o mesmo resultado, "
            "o que impede a detecção de mutações A1."
        )

        assert row["trip_count"] != row["total_fare"], (
            f"trip_count == total_fare para driver={row['driver_id']}. "
            "Mutação B (count→sum) pode não ser detectada."
        )
        assert row["trip_count"] != row["max_distance"], (
            f"trip_count == max_distance para driver={row['driver_id']}. "
            "Mutação B pode não ser detectada."
        )