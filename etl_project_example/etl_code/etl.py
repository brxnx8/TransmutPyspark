import os

from pyspark.sql import SparkSession

spark = SparkSession.builder\
  .master('local[*]')\
  .appName("Iniciando com Spark")\
  .getOrCreate()

"""#Importe das libs e funções necessarias"""

from pyspark.sql.functions import *
from pyspark.sql.window import Window
# from pyspark.sql.types import ArrayType, IntegerType

base_path = os.path.dirname(os.path.abspath(__file__))
    
file_path = os.path.join(base_path, "..", "data", "data.csv")

df_taxis = spark.read.csv(file_path, inferSchema=True, header=True, sep=',')

"""# Descrição das colunas do df_taxis e visualização das tabelas

**VendorID**: Um código indicando o provedor TPEP que forneceu o registro.

**tpep_pickup_datetime**: A data e hora em que o taxímetro foi acionado.

**tpep_dropoff_datetime**: A data e hora em que o taxímetro foi desligado.

**Passenger_count**: O número de passageiros no veículo.

**Trip_distance**: A distância da viagem em milhas relatada pelo taxímetro.

**PULocationID**: ID da Zona TLC na qual o taxímetro foi acionado.

**DOLocationID**: ID da Zona TLC na qual o taxímetro foi desligado.

**RateCodeID**: O código da tarifa em vigor no momento da viagem.
1 = Tarifa padrão;
2 = JFK;
3 = Newark;
4 = Nassau ou Westchester;
5 = Tarifa negociada;
6 = Grupo.

**Store_and_fwd_flag**: Este indicador mostra se o registro da viagem foi armazenado no veículo antes de ser enviado ao servidor.
Y = armazenar e encaminhar viagem
N = não armazenar e encaminhar viagem

**Payment_type**: Um código numérico que especifica como o passageiro pagou pela viagem.
1 = Crédito;
2 = Dinheiro;
3 = Sem cobrança;
4 = Disputa;
5 = Desconhecido;
6 = Viagem anulada (Voided trip).

**Fare_amount**: O valor da tarifa calculada pelo taxímetro.

**Extra**: Suplementos e encargos miscelâneos. Atualmente, isso inclui a sobretaxa de 0.50 à $1 para horas de pico e noturnas.

**MTA_tax**: Taxa de $0.50 que é automaticamente acionada pelo taxímetro.

**Improvement_surcharge**: Sobretaxa de $0.30 para viagens baseadas na taxa.

**Tip_amount**: O valor da gorjeta.

**Tolls_amount**: Total de pedágios pagos durante a viagem.

**Total_amount**: Total pago pelo passageiro.

**Congestion_Surcharge**: O total arrecadado pela taxa de congestionamento de NYC.

**Airport_fee**: $1.25 para retiradas nos aeroportos LaGuardia e John F. Kennedy.
"""

df_taxis.printSchema()


"""# 1-MTR (Substituição de Transformação de Mapeamento - Mapping Transformation Replacement)"""


def mtr_function(df):
    return (
        df.withColumn(
            "trip_duration_min",
            (unix_timestamp(col("tpep_dropoff_datetime")) -
             unix_timestamp(col("tpep_pickup_datetime"))) / 60
        )
        # O coalesce garante que se a coluna for nula, ele assume 0.0 para a soma não quebrar
        .withColumn(
            "valor_calculado",
            coalesce(col("fare_amount"), lit(0.0)) +
            coalesce(col("extra"), lit(0.0)) +
            coalesce(col("mta_tax"), lit(0.0)) +
            coalesce(col("tip_amount"), lit(0.0)) +
            coalesce(col("tolls_amount"), lit(0.0)) +
            coalesce(col("improvement_surcharge"), lit(0.0)) +
            coalesce(col("congestion_surcharge"), lit(0.0)) +
            coalesce(col("airport_fee"), lit(0.0))
        )
        # O when previne o erro de divisão por zero
        .withColumn(
            "valor_por_km",
            when(
                col("trip_distance") > 0,
                col("fare_amount") / col("trip_distance")
            ).otherwise(lit(None))
        )
    )


"""# 2-NFTP (Negação do Predicado de Transformação de Filtragem - Negation of Filter Transformation Predicate)"""


def nftp_function(df):
    return df.filter(
        col("trip_distance").isNotNull() &
        col("fare_amount").isNotNull() &
        (col("trip_distance") > 0.0) &
        (col("fare_amount") >= 2.50)
    )


"""# 3-ATR (Substituição de Transformação de Agregação - Aggregation Transformation Replacement)"""

def atr_fuction(rdd):
    """
    Calcula a receita média por KM para cada motorista.
    Utiliza o padrão otimizado de MapReduce para evitar gargalos de memória (Out Of Memory).
    """

    # 1. MAP: Prepara os dados.
    # Transforma cada linha em: (Chave, (Valor1, Valor2))
    # Ex: (Motorista_1, (10.0 dolares, 2.0 km))
    pares_iniciais = rdd.map(
        lambda row: (row["VendorID"], (row["fare_amount"], row["trip_distance"]))
    )

    # 2. REDUCE: A Agregação Distribuída (O ALVO DO ATR)
    # Soma as posições correspondentes das tuplas: (Soma_Dolares, Soma_Km)
    # A função lambda pega o acumulador 'a' e o novo valor 'b' e soma índice com índice.
    totais_agregados = pares_iniciais.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )

    # 3. MAP FINAL: O Cálculo da Média
    # Divide o total de dólares pelo total de KM (com proteção contra divisão por zero)
    resultado_final = totais_agregados.mapValues(
        lambda totais: totais[0] / totais[1] if totais[1] > 0 else 0.0
    )

    return resultado_final

"""# 4-UTS (Troca de Transformações Unárias - Unary Transformations Swap)"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile test_uts.py
# 
# import pytest
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
# from pyspark.sql.types import StructType, StructField, StringType, DoubleType
# from pyspark.testing.utils import assertDataFrameEqual
# 
# @pytest.fixture(scope="session")
# def spark():
#     # Inicialização limpa e moderna
#     spark_session = (SparkSession.builder
#         .master("local[1]")
#         .appName("pytest-pyspark-local")
#         .config("spark.ui.enabled", "false")
#         .getOrCreate())
# 
#     yield spark_session
#     spark_session.stop()
# 
# 
# def uts_function(df):
#     """
#     1. Unary Transformation A: Filtra corridas com tarifa > 50.0
#     2. Unary Transformation B: Adiciona taxa administrativa de 5.0
#     """
#     return (
#         df
#         # Transformação 1 (Filter)
#         .filter(col("fare_amount") > 50.0)
#         # Transformação 2 (Map/withColumn)
#         .withColumn("fare_amount", col("fare_amount") + 5.0)
#     )
# 
# 
# def test_uts_function_inversao_de_sequencia(spark):
#     # 1. Preparação (Arrange)
#     schema = StructType([
#         StructField("trip_id", StringType(), True),
#         StructField("fare_amount", DoubleType(), True)
#     ])
# 
#     input_data = [
#         # Original: 60.0 > 50 (Passa) -> 60.0 + 5.0 = 65.0
#         # Mutante: 60.0 + 5.0 = 65.0 -> 65.0 > 50 (Passa)
#         ("1", 60.0),
# 
#         # Tarifa = 48.0
#         # Original: 48.0 > 50 (FALSO -> DESCARTADO)
#         # Mutante: 48.0 + 5.0 = 53.0 -> 53.0 > 50 (VERDADEIRO -> SOBREVIVE)
#         ("2", 48.0),
# 
#         # Tarifa = 50.0
#         # Original: 50.0 > 50 (FALSO -> DESCARTADO)
#         # Mutante: 50.0 + 5.0 = 55.0 -> 55.0 > 50 (VERDADEIRO -> SOBREVIVE)
#         ("3", 50.0),
# 
#         # Tarifa = 40.0
#         # Original: 40.0 > 50 (FALSO -> DESCARTADO)
#         # Mutante: 40.0 + 5.0 = 45.0 -> 45.0 > 50 (FALSO -> DESCARTADO)
#         ("4", 40.0)
#     ]
# 
#     input_df = spark.createDataFrame(input_data, schema)
# 
#     expected_data = [
#         ("1", 65.0)
#     ]
# 
#     expected_df = spark.createDataFrame(expected_data, schema)
# 
#     result_df = uts_function(input_df)
# 
#     assertDataFrameEqual(result_df, expected_df)
