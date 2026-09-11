from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .build_features import build_daily_features
from .logging_utils import configure_logging
from .train_model import walk_forward_validate

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
    error = predictions["predicted_tmax_c"] - predictions["actual_tmax_c"]
    rounded_error = np.rint(predictions["predicted_tmax_c"]) - np.rint(predictions["actual_tmax_c"])
    return {
        "mae_c": float(error.abs().mean()),
        "rmse_c": float(np.sqrt(np.mean(error ** 2))),
        "bias_c": float(error.mean()),
        "exact_bucket_pct": float((rounded_error == 0).mean() * 100),
        "within_1c_bucket_pct": float((rounded_error.abs() <= 1).mean() * 100),
    }


def _baseline_metrics(daily: pd.DataFrame, prediction_dates: pd.Series) -> list[dict]:
    frame = daily.set_index("date").copy()
    frame["previous_day_tmax"] = frame["target_tmax_c"].shift(1)
    selected = frame.loc[pd.to_datetime(prediction_dates)]
    actual = selected["target_tmax_c"]
    rows = []
    for name, prediction in {
        "previous_day_tmax": selected["previous_day_tmax"],
        "max_observed_at_cutoff": selected["temp_max_so_far"],
    }.items():
        valid = prediction.notna() & actual.notna()
        error = prediction[valid] - actual[valid]
        rounded_error = np.rint(prediction[valid]) - np.rint(actual[valid])
        rows.append({
            "baseline": name,
            "mae_c": float(error.abs().mean()),
            "rmse_c": float(np.sqrt(np.mean(error ** 2))),
            "bias_c": float(error.mean()),
            "exact_bucket_pct": float((rounded_error == 0).mean() * 100),
            "within_1c_bucket_pct": float((rounded_error.abs() <= 1).mean() * 100),
        })
    return rows


def main() -> None:
    configure_logging()
    cfg = load_config()
    weather_path = ROOT / cfg["weather_csv"]
    if not weather_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {weather_path}")

    hourly = pd.read_csv(weather_path)
    cutoffs = [int(h) for h in cfg.get("cutoff_hours", [cfg["cutoff_hour"]])]
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    baseline_rows: list[dict] = []

    logger.info("Weather Temperature Edge Lab v0.3.0")
    logger.info("Multi-cutoff local: %s", ", ".join(f"{h:02d}:00" for h in cutoffs))

    for position, cutoff in enumerate(cutoffs, start=1):
        logger.info("=== Cutoff %02d:00 (%d/%d) ===", cutoff, position, len(cutoffs))
        daily = build_daily_features(hourly, cutoff_hour=cutoff, station_timezone=cfg["station_timezone"])
        validation = walk_forward_validate(
            daily,
            min_train_days=int(cfg["min_train_days"]),
            n_estimators=int(cfg["n_estimators"]),
            random_state=int(cfg["random_state"]),
            progress_every=int(cfg.get("progress_every", 25)),
        )
        predictions = validation.predictions.copy()
        predictions.insert(1, "cutoff_hour_local", cutoff)
        predictions.to_csv(output_dir / f"walk_forward_predictions_{cutoff:02d}h.csv", index=False)

        metrics = _metrics(predictions)
        summaries.append({"cutoff_hour_local": cutoff, "n_predictions": len(predictions), **metrics})
        for row in _baseline_metrics(daily, predictions["date"]):
            baseline_rows.append({"cutoff_hour_local": cutoff, **row})
        logger.info(
            "Cutoff %02dh | MAE %.3f C | exact %.1f%% | ±1C bucket %.1f%%",
            cutoff, metrics["mae_c"], metrics["exact_bucket_pct"], metrics["within_1c_bucket_pct"],
        )

    summary = pd.DataFrame(summaries).sort_values("cutoff_hour_local")
    baselines = pd.DataFrame(baseline_rows).sort_values(["cutoff_hour_local", "baseline"])
    summary.to_csv(output_dir / "multi_cutoff_summary.csv", index=False)
    baselines.to_csv(output_dir / "baseline_summary.csv", index=False)

    logger.info("=== Information curve ===")
    for row in summary.itertuples(index=False):
        logger.info("%02dh | MAE %.3f C | exact %.1f%% | ±1C %.1f%%", row.cutoff_hour_local, row.mae_c, row.exact_bucket_pct, row.within_1c_bucket_pct)
    logger.info("Resultados: %s", output_dir / "multi_cutoff_summary.csv")
    logger.info("Baselines: %s", output_dir / "baseline_summary.csv")
    logger.info("Execução v0.3.0 concluída")


if __name__ == "__main__":
    main()
