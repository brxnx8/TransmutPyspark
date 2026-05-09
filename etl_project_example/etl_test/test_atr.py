"""
Teste unitário para o source code ATR (Aggregation Transformation Replacement).

Estratégia de cobertura por caso de mutação
============================================

Caso A1 – troca de função de agregação (.agg explícito)
    Para cada função mutada o valor resultante muda:
    • sum("fare_amount")  → valores de total_fare distintos para cada grupo
    • avg("tip_amount")   → valores de avg_tip distintos para cada grupo
    • max("trip_distance")→ valores de max_distance distintos para cada grupo
    Os dados são cuidadosamente escolhidos para que sum ≠ avg ≠ max ≠ min ≠ count
    em cada grupo, matando qualquer troca de função.

Caso A2 – troca de coluna de entrada do .agg
    As colunas fare_amount, tip_amount e trip_distance têm distribuições
    completamente diferentes, portanto trocar a coluna muda o resultado
    em qualquer linha do expected.

Caso B – troca de shorthand groupBy().count()
    trip_count é verificado explicitamente por grupo; count ≠ sum/avg/max
    de qualquer coluna numérica nos dados escolhidos.

Caso C – troca de função de janela (rank)
    Os dados são projetados com empates DENTRO de cada região para que:
    • rank()        → atribui o MESMO número aos empatados, pula o próximo
    • dense_rank()  → atribui o MESMO número mas NÃO pula
    • row_number()  → nunca empata
    O expected captura o valor de rank() exato, matando qualquer outra
    função de janela.

Caso D – remoção de chave do groupBy
    Existem dois motoristas na mesma região com tarifas diferentes.
    Se uma chave for removida os grupos se fundem e os valores agregados
    mudam (ex.: sum de todo o grupo ≠ sum por motorista).
"""

import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, IntegerType
)
from pyspark.testing.utils import assertDataFrameEqual

from atr import atr_function


# ── Fixture Spark ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    spark_session = (SparkSession.builder
        .master("local[1]")
        .appName("pytest-pyspark-atr")
        .config("spark.ui.enabled", "false")
        .getOrCreate())
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
#
# Motorista D1 – região "Norte" – 3 corridas
#   fare: 10.0, 20.0, 30.0  → sum=60  avg=20  max=30  count=3
#   tip:   1.0,  2.0,  3.0  → sum= 6  avg= 2  max= 3  count=3
#   dist:  5.0, 10.0, 15.0  → sum=30  avg=10  max=15  count=3
#   Todas as estatísticas são diferentes entre si → mata A1 e A2.
#
# Motorista D2 – região "Norte" – 3 corridas
#   fare: 10.0, 20.0, 30.0  → sum=60  (mesmo total_fare que D1)
#   tip:   4.0,  5.0,  6.0  → avg=5.0 (diferente de D1=2.0) → empate em fare_rank!
#   dist:  2.0,  4.0,  6.0  → max=6.0 (diferente de D1=15.0)
#   Ter D1 e D2 com total_fare idêntico (60) na mesma região "Norte" cria
#   empate que distingue rank() vs dense_rank() vs row_number() → mata C.
#   Remover "driver_id" do groupBy funde D1 e D2 → sum=120 ≠ 60 → mata D.
#
# Motorista D3 – região "Sul" – 2 corridas
#   fare: 50.0, 70.0  → sum=120  avg=60  max=70  count=2
#   tip:   8.0, 10.0  → avg=9.0
#   dist: 20.0, 25.0  → max=25.0
#   Região diferente → partição independente da janela → fare_rank=1 (top da região).
#   Remover "region" do groupBy funde "Norte" com "Sul" → valores mudam → mata D.
#
INPUT_DATA = [
    # driver_id, region,  trip_distance, fare_amount, tip_amount
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
#
# D1 / Norte: total_fare=60.0, avg_tip=2.0, max_distance=15.0, trip_count=3
#   rank() dentro de "Norte" ordenado por total_fare desc:
#     D1=60, D2=60 → empate → ambos recebem rank=1 (rank pula o 2, vai para 3)
#
# D2 / Norte: total_fare=60.0, avg_tip=5.0, max_distance=6.0,  trip_count=3
#   rank=1 (empatado com D1)
#   dense_rank() daria 1 também, mas row_number() quebraria o empate → mata C
#   Para distinguir rank de dense_rank precisamos de um 3º grupo na mesma região.
#   → D3 é "Sul", não interfere. Para distinguir rank de dense_rank dentro de
#   "Norte" com apenas 2 grupos empatados: ambos rank=1 E dense_rank=1.
#   O que os diferencia é o comportamento quando há um grupo ABAIXO do empate.
#   Adicionamos D4 "Norte" com total_fare=30 para forçar rank=3 vs dense_rank=2.
#
# D4 / Norte: total_fare=30.0, avg_tip=7.0, max_distance=3.0, trip_count=1
#   rank()       → 3  (D1 e D2 ocupam posições 1 e 2, então D4 é 3ª)
#   dense_rank() → 2  (contagem contínua: D1/D2 = nível 1, D4 = nível 2)
#   row_number() → 3  (não há empate; depende da ordem interna → não determinístico)
#   Esse valor (3 vs 2) mata a mutação C.
#
# D3 / Sul: total_fare=120.0, avg_tip=9.0, max_distance=25.0, trip_count=2, rank=1
#
EXPECTED_DATA = [
    # driver_id, region, total_fare, avg_tip, max_distance, trip_count, fare_rank
    ("D1", "Norte", 60.0,  2.0, 15.0, 3, 1),
    ("D2", "Norte", 60.0,  5.0,  6.0, 3, 1),
    ("D3", "Sul",  120.0,  9.0, 25.0, 2, 1),
    ("D4", "Norte", 30.0,  7.0,  3.0, 1, 3),  # rank=3 (não dense_rank=2)
]

# Dados extras de D4 para incluir no input
_D4_INPUT = [
    ("D4", "Norte", 3.0, 30.0, 7.0),
]


# ── Teste principal ───────────────────────────────────────────────────────────

def test_atr_function_aggregations(spark):
    """
    Valida todos os valores agregados explicitamente.
    Mata mutações A1, A2, B, D e a maioria de C.
    """
    all_input = INPUT_DATA + _D4_INPUT
    input_df = spark.createDataFrame(all_input, INPUT_SCHEMA)

    result_df = atr_function(input_df)

    expected_df = spark.createDataFrame(EXPECTED_DATA, OUTPUT_SCHEMA)

    assertDataFrameEqual(result_df, expected_df)


def test_atr_groupby_keys_both_required(spark):
    """
    Verifica que o resultado é correto por (driver_id, region) e que
    remover qualquer uma das chaves do groupBy produz resultados errados.

    Mata mutação D: se driver_id for removido, D1 e D2 se fundem em um
    único grupo "Norte" com total_fare=150, não 60. Se region for removida,
    D3 se funde com os motoristas de "Norte".
    """
    all_input = INPUT_DATA + _D4_INPUT
    input_df = spark.createDataFrame(all_input, INPUT_SCHEMA)

    result_df = atr_function(input_df).cache()

    # Número de linhas deve ser exatamente 4 (uma por driver+region)
    assert result_df.count() == 4, (
        "Esperado 4 grupos distintos (D1/Norte, D2/Norte, D3/Sul, D4/Norte). "
        "Se uma chave do groupBy for removida os grupos se fundem."
    )

    # D1 e D2 devem aparecer como grupos SEPARADOS em "Norte"
    norte_rows = result_df.filter(col("region") == "Norte").orderBy("driver_id").collect()
    assert len(norte_rows) == 3, (
        "Esperados 3 grupos em 'Norte' (D1, D2, D4). "
        "Remoção de driver_id do groupBy funde esses grupos."
    )

    # Verificação de valores por grupo para D1
    d1 = next(r for r in norte_rows if r["driver_id"] == "D1")
    assert d1["total_fare"] == 60.0, f"D1 total_fare esperado 60.0, obtido {d1['total_fare']}"
    assert d1["trip_count"] == 3,    f"D1 trip_count esperado 3, obtido {d1['trip_count']}"

    # Verificação de valores por grupo para D2
    d2 = next(r for r in norte_rows if r["driver_id"] == "D2")
    assert d2["total_fare"] == 60.0, f"D2 total_fare esperado 60.0, obtido {d2['total_fare']}"
    assert d2["trip_count"] == 3,    f"D2 trip_count esperado 3, obtido {d2['trip_count']}"

    # D3 deve estar isolado em "Sul"
    sul_rows = result_df.filter(col("region") == "Sul").collect()
    assert len(sul_rows) == 1, (
        "Esperado exatamente 1 grupo em 'Sul' (D3). "
        "Remoção de region do groupBy pode fundir regiões."
    )
    d3 = sul_rows[0]
    assert d3["total_fare"] == 120.0, f"D3 total_fare esperado 120.0, obtido {d3['total_fare']}"


def test_atr_window_rank_vs_dense_rank(spark):
    """
    Verifica explicitamente que rank() (e não dense_rank ou row_number) é usado.

    Com D1, D2 empatados em total_fare=60 e D4 em total_fare=30 dentro de "Norte":
      rank()       → D4 recebe rank=3  (pula a posição 2)
      dense_rank() → D4 recebe rank=2  (sem pular)
      row_number() → D4 recebe rank=3, mas D1 e D2 não teriam o mesmo número

    Mata mutação C.
    """
    all_input = INPUT_DATA + _D4_INPUT
    input_df = spark.createDataFrame(all_input, INPUT_SCHEMA)

    result_df = atr_function(input_df)

    norte = result_df.filter(col("region") == "Norte").orderBy("driver_id").collect()
    d1 = next(r for r in norte if r["driver_id"] == "D1")
    d2 = next(r for r in norte if r["driver_id"] == "D2")
    d4 = next(r for r in norte if r["driver_id"] == "D4")

    # D1 e D2 empatados → ambos rank=1
    assert d1["fare_rank"] == 1, (
        f"D1 fare_rank esperado 1 (empatado com D2), obtido {d1['fare_rank']}. "
        "Indica que a função de janela foi trocada."
    )
    assert d2["fare_rank"] == 1, (
        f"D2 fare_rank esperado 1 (empatado com D1), obtido {d2['fare_rank']}. "
        "Indica que a função de janela foi trocada."
    )

    # D4 deve ter rank=3, não rank=2 (o que dense_rank produziria)
    assert d4["fare_rank"] == 3, (
        f"D4 fare_rank esperado 3 com rank(), obtido {d4['fare_rank']}. "
        "Se o valor for 2, a função foi mutada para dense_rank(). "
        "Se D1 e D2 não tiverem o mesmo rank, pode ser row_number()."
    )


def test_atr_aggregation_values_are_distinct(spark):
    """
    Garante que os valores de total_fare, avg_tip e max_distance são
    numericamente distintos entre si em cada grupo, tornando impossível
    confundir uma função de agregação com outra (mata A1).

    Também verifica que trip_count (resultado do shorthand .count()) é
    diferente de qualquer valor agregado numérico (mata B).
    """
    all_input = INPUT_DATA + _D4_INPUT
    input_df = spark.createDataFrame(all_input, INPUT_SCHEMA)

    result_df = atr_function(input_df)

    for row in result_df.collect():
        vals = {
            "total_fare":   row["total_fare"],
            "avg_tip":      row["avg_tip"],
            "max_distance": row["max_distance"],
        }
        # Todos os valores de agregação devem ser distintos entre si
        assert len(set(vals.values())) == len(vals), (
            f"Valores de agregação não são todos distintos para "
            f"driver={row['driver_id']} region={row['region']}: {vals}. "
            "Isso indica que duas funções de agregação produzem o mesmo resultado, "
            "o que impede a detecção de mutações A1."
        )

        # trip_count deve ser diferente de total_fare e max_distance
        assert row["trip_count"] != row["total_fare"], (
            f"trip_count == total_fare para driver={row['driver_id']}. "
            "Mutação B (count→sum) pode não ser detectada."
        )
        assert row["trip_count"] != row["max_distance"], (
            f"trip_count == max_distance para driver={row['driver_id']}. "
            "Mutação B pode não ser detectada."
        )