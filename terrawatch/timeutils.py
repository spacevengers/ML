"""Acquisition-time helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_minutes(acq_date, acq_time) -> np.ndarray:
    """FIRMS ``acq_date`` (YYYY-MM-DD) + ``acq_time`` (HHMM) -> minutes since epoch.

    FIRMS stores time as an integer HHMM, so 08:31 arrives as ``831``. Padding is
    required, otherwise the hour/minute split is wrong.
    """
    date_s = pd.Series(np.asarray(acq_date))
    time_s = pd.Series(np.asarray(acq_time)).astype("int64").astype(str).str.zfill(4)

    days = (pd.to_datetime(date_s) - pd.Timestamp("1970-01-01")).dt.days
    hours = time_s.str[:2].astype("int64")
    minutes = time_s.str[2:].astype("int64")

    return (days * 1440 + hours * 60 + minutes).to_numpy()


def datetime_from(acq_date, acq_time) -> pd.Series:
    """Combine FIRMS date + time into a real timestamp."""
    date_s = pd.Series(np.asarray(acq_date))
    time_s = pd.Series(np.asarray(acq_time)).astype("int64").astype(str).str.zfill(4)
    return pd.to_datetime(date_s + " " + time_s.str[:2] + ":" + time_s.str[2:])
