
import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
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





def test_atr_function(spark):
    # O RDD pertence ao SparkContext, não à Session principal
    sc = spark.sparkContext

    # Arrange: Dados estrategicamente escolhidos
    data = [
        # Cenário 1: Múltiplas corridas para o mesmo motorista (V1)
        # Corrida 1: $10 / 2km (Média isolada: 5.0)
        Row(VendorID="V1", fare_amount=10.0, trip_distance=2.0),
        # Corrida 2: $26 / 4km (Média isolada: 6.5)
        Row(VendorID="V1", fare_amount=26.0, trip_distance=4.0),
        # Média Global Esperada V1: ($10 + $26) / (2km + 4km) = $36 / 6km = 6.0

        # Cenário 2: Motorista com distância ZERO (Edge Case para a divisão)
        Row(VendorID="V2", fare_amount=15.0, trip_distance=0.0),

        # Cenário 3: Motorista com apenas UMA corrida (Garante que chaves únicas não quebram)
        Row(VendorID="V3", fare_amount=20.0, trip_distance=5.0)
    ]

    rdd_input = sc.parallelize(data)

    # Act
    rdd_output = atr_fuction(rdd_input)

    # Convertendo o RDD final (que é uma lista de tuplas) em um Dicionário Python
    # Isso transforma [('V1', 6.0), ('V2', 0.0)] em {'V1': 6.0, 'V2': 0.0}
    resultados = dict(rdd_output.collect())

    # Assert
    # 1. Valida se todos os motoristas foram processados
    assert len(resultados) == 3, "Nem todos os motoristas retornaram no processamento"

    # 2. Valida a matemática da agregação (O executor de mutantes ATR)
    assert resultados["V1"] == 6.0, "Falha na agregação de múltiplas corridas"

    # 3. Valida a proteção contra divisão por zero
    assert resultados["V2"] == 0.0, "Falha no tratamento de distância zero"

    # 4. Valida motoristas com corrida única
    assert resultados["V3"] == 4.0, "Falha no processamento de corrida única"