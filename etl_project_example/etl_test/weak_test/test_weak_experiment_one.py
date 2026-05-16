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

def test_valid_trips_with_tip_rate_weak(spark):
    df = _make_valid_trips_df(spark)
    result = valid_trips_with_tip_rate(df)

    assert result.count() > 0
    assert "tip_rate" in result.columns

def test_revenue_by_vendor_weak(spark):
    df = _make_revenue_df(spark)
    result = revenue_by_vendor(df)

    assert result.count() > 0
    assert "total_revenue" in result.columns
    assert "trip_count" in result.columns

def test_trip_profile_by_zone_weak(spark):
    df = _make_zone_df(spark)
    result = trip_profile_by_zone(df)

    assert result.count() > 0
    for col in ["avg_duration_min", "avg_distance_miles", "max_speed_mph", "trip_count"]:
        assert col in result.columns

def test_tip_ranking_by_hour_weak(spark):
    df = _make_tip_ranking_df(spark)
    result = tip_ranking_by_hour(df)

    assert result.count() > 0
    assert "tip_rank" in result.columns
    assert "pickup_hour" in result.columns                