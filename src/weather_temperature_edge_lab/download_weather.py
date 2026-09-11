from __future__ import annotations

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .logging_utils import configure_logging

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def download_iem_asos(station: str, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = [
        ("station", station),
        ("data", "tmpc"),
        ("data", "relh"),
        ("data", "sknt"),
        ("data", "drct"),
        ("data", "mslp"),
        ("data", "p01m"),
        ("year1", str(start_ts.year)),
        ("month1", str(start_ts.month)),
        ("day1", str(start_ts.day)),
        ("year2", str(end_ts.year)),
        ("month2", str(end_ts.month)),
        ("day2", str(end_ts.day)),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "empty"),
        ("trace", "empty"),
        ("direct", "no"),
        ("report_type", "1"),
        ("report_type", "2"),
    ]

    logger.info("Baixando histórico da estação %s", station)
    logger.info("Período solicitado: %s até %s", start, end)

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    text = response.text.strip()

    if not text or "station,valid" not in text.lower():
        raise RuntimeError("A fonte não retornou dados válidos para a estação informada.")

    raw = pd.read_csv(StringIO(text))
    if raw.empty:
        raise RuntimeError(f"Nenhum dado retornado para a estação {station}.")

    raw = raw.rename(
        columns={
            "valid": "timestamp",
            "tmpc": "temp_c",
            "relh": "humidity",
            "drct": "wind_dir_deg",
            "mslp": "pressure_hpa",
            "p01m": "precip_mm",
        }
    )

    raw["wind_speed_kmh"] = pd.to_numeric(raw.get("sknt"), errors="coerce") * 1.852
    raw["cloud_cover_pct"] = pd.NA

    for column in ["temp_c", "humidity", "wind_dir_deg", "pressure_hpa", "precip_mm"]:
        if column not in raw.columns:
            raw[column] = pd.NA
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["timestamp"] = (
        pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    keep = [
        "timestamp",
        "temp_c",
        "humidity",
        "wind_speed_kmh",
        "wind_dir_deg",
        "pressure_hpa",
        "cloud_cover_pct",
        "precip_mm",
    ]

    out = raw[keep].dropna(subset=["timestamp", "temp_c"]).copy()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    logger.info("Registros válidos baixados: %d", len(out))
    if not out.empty:
        logger.info("Cobertura real: %s até %s", out["timestamp"].min(), out["timestamp"].max())

    return out


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Baixa histórico horário meteorológico.")
    parser.add_argument("--station", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "weather_hourly.csv"),
    )
    args = parser.parse_args()

    df = download_iem_asos(args.station, args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Arquivo salvo em: %s", output)


if __name__ == "__main__":
    main()
