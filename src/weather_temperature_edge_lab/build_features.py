from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "timestamp",
    "temp_c",
    "humidity",
    "wind_speed_kmh",
    "wind_dir_deg",
    "pressure_hpa",
    "cloud_cover_pct",
    "precip_mm",
]


def _last_valid(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def _delta_from_hours(df: pd.DataFrame, hours: int) -> float:
    values = df["temp_c"].dropna()
    if len(values) < 2:
        return np.nan

    last_time = df.loc[values.index[-1], "timestamp"]
    target_time = last_time - pd.Timedelta(hours=hours)
    previous = df[df["timestamp"] <= target_time]["temp_c"].dropna()
    if previous.empty:
        return np.nan
    return float(values.iloc[-1] - previous.iloc[-1])


def build_daily_features(hourly: pd.DataFrame, cutoff_hour: int = 10) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in hourly.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    df = hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    logger.info("Construindo features diárias com cutoff às %02dh", cutoff_hour)

    rows: list[dict] = []
    for date, day in df.groupby("date", sort=True):
        observed = day[day["hour"] <= cutoff_hour].sort_values("timestamp")
        if observed.empty or day["temp_c"].dropna().empty:
            continue

        ts = observed["timestamp"].iloc[-1]
        doy = ts.dayofyear
        wind_direction = _last_valid(observed["wind_dir_deg"])
        wind_direction_rad = (
            np.deg2rad(wind_direction) if not np.isnan(wind_direction) else np.nan
        )
        pressure = observed["pressure_hpa"].dropna()

        rows.append(
            {
                "date": pd.Timestamp(date),
                "target_tmax_c": float(day["temp_c"].max()),
                "temp_last": _last_valid(observed["temp_c"]),
                "temp_mean": float(observed["temp_c"].mean()),
                "temp_min": float(observed["temp_c"].min()),
                "temp_max_so_far": float(observed["temp_c"].max()),
                "temp_std": float(observed["temp_c"].std(ddof=0)),
                "temp_range_so_far": float(
                    observed["temp_c"].max() - observed["temp_c"].min()
                ),
                "temp_delta_1h": _delta_from_hours(observed, 1),
                "temp_delta_3h": _delta_from_hours(observed, 3),
                "temp_delta_6h": _delta_from_hours(observed, 6),
                "humidity_last": _last_valid(observed["humidity"]),
                "humidity_mean": float(observed["humidity"].mean()),
                "pressure_last": _last_valid(observed["pressure_hpa"]),
                "pressure_delta": (
                    float(pressure.iloc[-1] - pressure.iloc[0])
                    if len(pressure) >= 2
                    else np.nan
                ),
                "wind_speed_last": _last_valid(observed["wind_speed_kmh"]),
                "wind_speed_mean": float(observed["wind_speed_kmh"].mean()),
                "wind_dir_sin": (
                    float(np.sin(wind_direction_rad))
                    if not np.isnan(wind_direction_rad)
                    else np.nan
                ),
                "wind_dir_cos": (
                    float(np.cos(wind_direction_rad))
                    if not np.isnan(wind_direction_rad)
                    else np.nan
                ),
                "cloud_last": _last_valid(observed["cloud_cover_pct"]),
                "cloud_mean": float(observed["cloud_cover_pct"].mean()),
                "precip_sum": float(observed["precip_mm"].sum()),
                "month": ts.month,
                "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
                "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
            }
        )

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    logger.info("Dias válidos construídos: %d", len(daily))
    return daily
