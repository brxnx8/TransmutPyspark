from etl_project_example.etl_test.conftest import (
    _make_efficiency_df
)    

from etl_project_example.etl_code.experiment_two import (
    full_trip_efficiency_analysis,
)

def test_full_trip_efficiency_analysis_weak(spark):
    df = _make_efficiency_df(spark)
    result = full_trip_efficiency_analysis(df)

    assert result.count() > 0
    for col in ["PULocationID", "shift", "total_fare",
                "avg_distance", "avg_cost_per_mile", "trip_count"]:
        assert col in result.columns