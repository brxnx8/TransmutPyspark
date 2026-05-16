from datetime import datetime

from etl_project_example.etl_test.conftest import (
    _make_valid_trips_df, 
    _make_revenue_df,
    _make_zone_df,
    _make_tip_ranking_df,
)    

from etl_project_example.etl_code.experiment_one import (
    valid_trips_with_tip_rate,
    revenue_by_vendor,
    trip_profile_by_zone,
    tip_ranking_by_hour,
)

def test_valid_trips_with_tip_rate_strong(spark):
    df = _make_valid_trips_df(spark)
    result = valid_trips_with_tip_rate(df).collect()

    assert len(result) == 2, "apenas as 2 corridas payment_type=1 e distance>0 devem sobrar"

    tip_rates = sorted(r["tip_rate"] for r in result)
    assert abs(tip_rates[0] - 0.1) < 1e-9
    assert abs(tip_rates[1] - 0.1) < 1e-9

    # NFTP: payment_type == 1 — verifica que payment_type=2 é excluído
    pt_values = {r["payment_type"] for r in result}
    assert pt_values == {1}

    # NFTP: trip_distance > 0.0 — verifica que distance=0 e distance<0 são excluídos
    for r in result:
        assert r["trip_distance"] > 0.0

    # MTR: tip_rate = tip_amount / total_amount, não tip_amount / trip_distance etc.
    for r in result:
        expected = r["tip_amount"] / r["total_amount"]
        assert abs(r["tip_rate"] - expected) < 1e-9

    # MTR: coluna tip_rate existe e as outras colunas originais permanecem
    cols = result[0].asDict().keys()
    assert "tip_rate" in cols
    assert "payment_type" in cols

    # Fronteira exata do filtro: distance=0.0 deve ser excluída (operador >, não >=)
    boundary_df = spark.createDataFrame(
        [(1, 0.0, 0.0, 5.0)],
        schema=["payment_type", "trip_distance", "tip_amount", "total_amount"],
    )
    assert valid_trips_with_tip_rate(boundary_df).count() == 0

    # Fronteira exata do filtro: payment_type=0 e payment_type=2 excluídos
    other_pt_df = spark.createDataFrame(
        [(0, 1.0, 1.0, 5.0), (2, 1.0, 1.0, 5.0)],
        schema=["payment_type", "trip_distance", "tip_amount", "total_amount"],
    )
    assert valid_trips_with_tip_rate(other_pt_df).count() == 0

def test_revenue_by_vendor_strong(spark):
    df = _make_revenue_df(spark)
    result = {r["VendorID"]: r for r in revenue_by_vendor(df).collect()}

    assert set(result.keys()) == {1, 2}

    # ATR: sum correto, não count ou avg
    assert abs(result[1]["total_revenue"] - 30.0) < 1e-9
    assert abs(result[2]["total_revenue"] -  5.0) < 1e-9

    # ATR: count correto
    assert result[1]["trip_count"] == 2
    assert result[2]["trip_count"] == 1

    # NFTP: total_amount <= 0 filtrado (zero e negativo)
    for r in result.values():
        assert r["total_revenue"] > 0.0

    # MTR: select mantém exatamente as 3 colunas esperadas
    cols = set(revenue_by_vendor(df).columns)
    assert cols == {"VendorID", "total_revenue", "trip_count"}

    # NFTP fronteira: valor exatamente 0.0 deve ser excluído
    zero_df = spark.createDataFrame([(1, 0.0)], ["VendorID", "total_amount"])
    assert revenue_by_vendor(zero_df).count() == 0

    # NFTP fronteira: valor 0.001 deve ser incluído
    tiny_df = spark.createDataFrame([(1, 0.001)], ["VendorID", "total_amount"])
    assert revenue_by_vendor(tiny_df).count() == 1

def test_trip_profile_by_zone_strong(spark):
    df = _make_zone_df(spark)
    result = {r["PULocationID"]: r for r in trip_profile_by_zone(df).collect()}

    # apenas zonas 1 e 2 sobrevivem
    assert set(result.keys()) == {1, 2}

    # ATR: avg_duration_min correto (30 min)
    assert abs(result[1]["avg_duration_min"] - 30.0) < 1e-6
    # ATR: avg_distance_miles correto (3.0)
    assert abs(result[1]["avg_distance_miles"] - 3.0) < 1e-9
    # ATR: count correto
    assert result[1]["trip_count"] == 2
    # ATR: max_speed_mph é max, não avg
    assert abs(result[2]["max_speed_mph"] - 60.0) < 1e-4

    # MTR: trip_duration_min = diferença em minutos (não segundos, não horas)
    # Uma corrida de 30 min deve resultar em avg_duration_min ≈ 30
    assert result[1]["avg_duration_min"] < 60.0

    # MTR: speed_mph = distance / (duration_min / 60)
    # zona 2: 60 milhas / 1 hora = 60 mph
    assert abs(result[2]["max_speed_mph"] - 60.0) < 1e-4

    # NFTP: passenger_count=0 filtrado
    assert 3 not in result

    # NFTP: trip_distance=0 filtrado
    zero_dist_df = spark.createDataFrame(
        [(99, 1, 0.0, datetime(2023,1,1,8,0), datetime(2023,1,1,8,30))],
        schema=["PULocationID","passenger_count","trip_distance",
                "tpep_pickup_datetime","tpep_dropoff_datetime"],
    )
    assert trip_profile_by_zone(zero_dist_df).count() == 0

    # NFTP: corridas com speed >= 100 mph filtradas
    fast_df = spark.createDataFrame(
        # 200 milhas em 60 min = 200 mph → deve ser filtrado
        [(99, 1, 200.0, datetime(2023,1,1,8,0), datetime(2023,1,1,9,0))],
        schema=["PULocationID","passenger_count","trip_distance",
                "tpep_pickup_datetime","tpep_dropoff_datetime"],
    )
    assert trip_profile_by_zone(fast_df).count() == 0

    # NFTP fronteira speed: 99 mph incluído, 101 mph excluído
    ok_df = spark.createDataFrame(
        [(99, 1, 99.0, datetime(2023,1,1,8,0), datetime(2023,1,1,9,0))],
        schema=["PULocationID","passenger_count","trip_distance",
                "tpep_pickup_datetime","tpep_dropoff_datetime"],
    )
    assert trip_profile_by_zone(ok_df).count() == 1

def test_tip_ranking_by_hour_strong(spark):
    df = _make_tip_ranking_df(spark)
    result = tip_ranking_by_hour(df).collect()

    by_hour = {}
    for r in result:
        by_hour.setdefault(r["pickup_hour"], []).append(r)

    # hora 8: exatamente 3 corridas (top-3)
    assert len(by_hour[8]) == 3

    # ATR: rank() — rank 1 tem a maior gorjeta
    tips_h8 = sorted(
        [(r["tip_rank"], r["tip_amount"]) for r in by_hour[8]]
    )
    assert tips_h8[0] == (1, 9.0)
    assert tips_h8[1] == (2, 7.0)
    assert tips_h8[2] == (3, 5.0)

    # hora 10: apenas 2 corridas, ambas com rank <= 3
    assert len(by_hour[10]) == 2

    # NFTP: gorjeta = 0.0 excluída
    all_tips = [r["tip_amount"] for r in result]
    assert all(t > 0.0 for t in all_tips)

    # NFTP: where tip_rank <= 3 — nenhuma linha com rank > 3
    all_ranks = [r["tip_rank"] for r in result]
    assert max(all_ranks) <= 3
    assert 4 not in all_ranks

    # MTR: pickup_hour vem de hour(tpep_pickup_datetime)
    assert all(r["pickup_hour"] in {8, 10} for r in result)

    # MTR: select contém exatamente as colunas certas
    assert set(tip_ranking_by_hour(df).columns) == {
        "pickup_hour", "tip_amount", "tip_rank", "VendorID"
    }

    # NFTP fronteira: tip_amount = 0.0 excluído
    zero_df = spark.createDataFrame(
        [(1, 0.0, datetime(2023,1,1,8,0,0))],
        schema=["VendorID","tip_amount","tpep_pickup_datetime"],
    )
    assert tip_ranking_by_hour(zero_df).count() == 0        