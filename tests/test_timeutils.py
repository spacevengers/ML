import pandas as pd

from terrawatch.timeutils import datetime_from, to_minutes


def test_hhmm_is_padded_so_0831_is_not_831_minutes():
    # 08:31 -> 8*60 + 31 = 511 minutes after midnight
    assert to_minutes(["1970-01-01"], [831])[0] == 511


def test_midnight_is_zero():
    assert to_minutes(["1970-01-01"], [0])[0] == 0


def test_days_accumulate():
    one = to_minutes(["1970-01-01"], [1200])[0]
    two = to_minutes(["1970-01-02"], [1200])[0]
    assert two - one == 1440


def test_ordering_is_preserved_within_a_day():
    values = to_minutes(["2023-02-10"] * 3, [0, 1200, 2359])
    assert values[0] < values[1] < values[2]


def test_midnight_boundary_is_two_minutes_apart():
    before = to_minutes(["2023-02-10"], [2359])[0]
    after = to_minutes(["2023-02-11"], [1])[0]
    assert after - before == 2


def test_datetime_from_combines_fields():
    out = datetime_from(pd.Series(["2023-02-10"]), pd.Series([831]))
    assert out.iloc[0] == pd.Timestamp("2023-02-10 08:31")
