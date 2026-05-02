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
    # trip_id | trip_distance | fare_amount | vendor_id
    ("1",   5.0,   15.0,  "A"),   # ✔ sobrevive ao filter (fare >= 2.5) → entra no orderBy/limit
    ("2",   1.0,    2.50, "B"),   # ✔ sobrevive
    ("3",   0.0,   10.0,  "A"),   # ✔ sobrevive
    ("4",  -1.5,   12.0,  "C"),   # ✔ sobrevive
    ("5",   2.0,    2.49, "A"),   # ✗ eliminado pelo filter (fare < 2.5)
    ("6",   3.0,    8.0,  "B"),   # ✔ sobrevive
    ("7",   4.0,   20.0,  "C"),   # ✔ sobrevive
    ("8",   6.0,    5.0,  "A"),   # ✔ sobrevive
]

# Após filter(fare >= 2.5) + select(...) temos as linhas sem trip_id=5
# Após withColumnRenamed(vendor_id→provider_id) + drop(provider_id)
#   → coluna vendor_id / provider_id desaparece; ficamos com 3 colunas
# Após orderBy(fare_amount desc) + limit(3)
#   → top-3 por fare_amount: trip_id 7 (20.0), 1 (15.0), 4 (12.0)

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


# ---------------------------------------------------------------------------
# Teste 1 — comportamento correto do pipeline original
# ---------------------------------------------------------------------------
def test_uts_pipeline_result(spark):
    """
    Verifica que o pipeline produz exatamente os 3 registros esperados
    após: filter → select → withColumnRenamed → drop → orderBy → limit.
    """
    input_df    = spark.createDataFrame(INPUT_DATA, SCHEMA)
    expected_df = spark.createDataFrame(EXPECTED_DATA, SCHEMA_EXPECTED)

    result_df = uts_function(input_df)

    assertDataFrameEqual(result_df, expected_df)


# ---------------------------------------------------------------------------
# Teste 2 — o filtro é aplicado ANTES do select (ordem importa)
# Garante que um mutante que inverta select↔filter produza resultado diferente
# ---------------------------------------------------------------------------
def test_uts_filter_before_select_matters(spark):
    """
    Se o mutante trocar select→filter para filter→select, o select tentará
    projetar colunas que podem não existir (ou o resultado mudará).
    Aqui verificamos que a função original mantém as colunas corretas.
    """
    input_df = spark.createDataFrame(INPUT_DATA, SCHEMA)
    result_df = uts_function(input_df)

    # Deve conter exatamente 3 colunas (vendor_id foi renomeada e dropada)
    assert result_df.columns == ["trip_id", "trip_distance", "fare_amount"], (
        "Pipeline deve entregar apenas trip_id, trip_distance e fare_amount"
    )


# ---------------------------------------------------------------------------
# Teste 3 — a linha com fare_amount < 2.5 nunca aparece no resultado
# ---------------------------------------------------------------------------
def test_uts_filtered_row_absent(spark):
    """
    trip_id='5' tem fare_amount=2.49 e deve ser eliminada pelo filter.
    Um mutante que aplique limit antes de orderBy poderia incluí-la ou
    alterar quais linhas chegam ao topo; o original jamais a inclui.
    """
    input_df  = spark.createDataFrame(INPUT_DATA, SCHEMA)
    result_df = uts_function(input_df)

    trip_ids = [row.trip_id for row in result_df.collect()]
    assert "5" not in trip_ids, (
        "trip_id='5' (fare_amount=2.49) deve ter sido removida pelo filter"
    )


# ---------------------------------------------------------------------------
# Teste 4 — limit respeita a ordenação (orderBy vem antes de limit)
# ---------------------------------------------------------------------------
def test_uts_limit_respects_order(spark):
    """
    orderBy(fare_amount desc) deve ocorrer ANTES de limit(3).
    O resultado deve ser os 3 maiores fare_amounts disponíveis após o filter.

    Um mutante limit→orderBy aplicaria limit em ordem arbitrária e
    então ordenaria — poderia devolver 3 registros diferentes.
    """
    input_df  = spark.createDataFrame(INPUT_DATA, SCHEMA)
    result_df = uts_function(input_df)

    fares = [row.fare_amount for row in result_df.collect()]

    # Os 3 maiores fare_amounts após filtro são: 20.0, 15.0, 12.0
    assert sorted(fares, reverse=True) == [20.0, 15.0, 12.0], (
        "limit(3) deve pegar os 3 maiores fare_amounts (orderBy aplicado antes)"
    )


# ---------------------------------------------------------------------------
# Teste 5 — número de linhas retornadas é exatamente 3
# ---------------------------------------------------------------------------
def test_uts_row_count(spark):
    """
    limit(3) deve garantir exatamente 3 linhas no resultado final,
    independente do tamanho do DataFrame de entrada.
    """
    input_df  = spark.createDataFrame(INPUT_DATA, SCHEMA)
    result_df = uts_function(input_df)

    assert result_df.count() == 3, "Pipeline deve retornar exatamente 3 linhas"