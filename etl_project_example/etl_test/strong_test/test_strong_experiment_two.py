from datetime import datetime

from etl_project_example.etl_test.conftest import (
    _make_efficiency_df
)    

from etl_project_example.etl_code.experiment_two import (
    full_trip_efficiency_analysis,
)

def test_full_trip_efficiency_analysis_strong(spark):
    df = _make_efficiency_df(spark)
    result = full_trip_efficiency_analysis(df)
    rows = {(r["PULocationID"], r["shift"]): r for r in result.collect()}

    # zonas filtradas não aparecem
    assert all(r[0] != 3 for r in rows)

    # MTR: shift correto por faixa de hora
    assert (1, "afternoon") in rows
    assert (2, "overnight") in rows

    # ATR: sum(fare_amount) correto por grupo
    assert abs(rows[(1, "afternoon")]["total_fare"] - 450.0) < 1e-6

    # ATR: avg(trip_distance) correto
    assert abs(rows[(1, "afternoon")]["avg_distance"] - 30.0) < 1e-9

    # ATR: count correto
    assert rows[(1, "afternoon")]["trip_count"] == 3

    # ATR: avg(cost_per_mile) = fare/distance = 5.0
    assert abs(rows[(1, "afternoon")]["avg_cost_per_mile"] - 5.0) < 1e-6

    # MTR: select contém exatamente as colunas corretas
    expected_cols = {"PULocationID", "shift", "total_fare",
                     "avg_distance", "avg_cost_per_mile", "trip_count"}
    assert set(result.columns) == expected_cols

    # NFTP: fare_amount <= 0 filtrado
    neg_df = spark.createDataFrame(
        [(1, -1.0, 5.0, 1,
          datetime(2023,1,1,15,0), datetime(2023,1,1,16,0))],
        schema=["PULocationID","fare_amount","trip_distance","passenger_count",
                "tpep_pickup_datetime","tpep_dropoff_datetime"],
    )
    assert full_trip_efficiency_analysis(neg_df).count() == 0

    # NFTP: passenger_count = 0 filtrado
    pax_df = spark.createDataFrame(
        [(1, 10.0, 5.0, 0,
          datetime(2023,1,1,15,0), datetime(2023,1,1,16,0))],
        schema=["PULocationID","fare_amount","trip_distance","passenger_count",
                "tpep_pickup_datetime","tpep_dropoff_datetime"],
    )
    assert full_trip_efficiency_analysis(pax_df).count() == 0

    # MTR: shift "morning" 6h-11h59
    morn_df = spark.createDataFrame(
        [(1, 10.0, 5.0, 1,
          datetime(2023,1,1,6,0), datetime(2023,1,1,7,0))],
        schema=["PULocationID","fare_amount","trip_distance","passenger_count",
                "tpep_pickup_datetime","tpep_dropoff_datetime"],
    )
    morn_result = full_trip_efficiency_analysis(morn_df).collect()
    assert len(morn_result) == 1
    assert morn_result[0]["shift"] == "morning"

    # MTR: shift "night" 18h-23h59
    night_df = spark.createDataFrame(
        [(1, 10.0, 5.0, 1,
          datetime(2023,1,1,20,0), datetime(2023,1,1,21,0))],
        schema=["PULocationID","fare_amount","trip_distance","passenger_count",
                "tpep_pickup_datetime","tpep_dropoff_datetime"],
    )
    night_result = full_trip_efficiency_analysis(night_df).collect()
    assert len(night_result) == 1
    assert night_result[0]["shift"] == "night"

    # MTR: shift "overnight" 0h-5h59
    ovn_df = spark.createDataFrame(
        [(1, 10.0, 5.0, 1,
          datetime(2023,1,1,3,0), datetime(2023,1,1,3,30))],
        schema=["PULocationID","fare_amount","trip_distance","passenger_count",
                "tpep_pickup_datetime","tpep_dropoff_datetime"],
    )
    ovn_result = full_trip_efficiency_analysis(ovn_df).collect()
    assert len(ovn_result) == 1
    assert ovn_result[0]["shift"] == "overnight"

    # ATR: groupBy por (PULocationID, shift) — não apenas por PULocationID
    assert (1, "afternoon") in rows
    # se fosse só por zona, não haveria chave composta
    zone1_rows = [k for k in rows if k[0] == 1]
    assert len(zone1_rows) >= 1