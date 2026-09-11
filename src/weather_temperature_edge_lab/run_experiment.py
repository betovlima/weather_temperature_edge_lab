from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .build_features import build_daily_features
from .logging_utils import configure_logging
from .plot_results import save_prediction_chart
from .score_market import score_market
from .train_model import fit_final_and_distribution, walk_forward_validate

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def main() -> None:
    configure_logging()
    cfg = load_config()

    weather_path = ROOT / cfg["weather_csv"]
    market_path = ROOT / cfg["market_csv"]
    output_path = ROOT / cfg["output_csv"]
    chart_path = ROOT / "outputs" / "predicted_vs_actual.png"
    validation_csv = ROOT / "outputs" / "walk_forward_predictions.csv"

    logger.info("Weather Temperature Edge Lab")
    logger.info("Carregando histórico: %s", weather_path)

    if not weather_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {weather_path}\n"
            "Execute antes: poetry run weather-download --station ZUUU --start 2021-01-01 --end 2026-09-11"
        )

    hourly = pd.read_csv(weather_path)
    logger.info("Registros horários carregados: %d", len(hourly))
    if not hourly.empty:
        timestamps = pd.to_datetime(hourly["timestamp"])
        logger.info("Período disponível: %s até %s", timestamps.min(), timestamps.max())

    daily = build_daily_features(hourly, cutoff_hour=int(cfg["cutoff_hour"]))
    if len(daily) <= int(cfg["min_train_days"]):
        raise ValueError(
            f"Dias válidos insuficientes: {len(daily)}. "
            f"São necessários mais de {cfg['min_train_days']} dias."
        )

    validation = walk_forward_validate(
        daily,
        min_train_days=int(cfg["min_train_days"]),
        n_estimators=int(cfg["n_estimators"]),
        random_state=int(cfg["random_state"]),
        progress_every=int(cfg.get("progress_every", 25)),
    )

    validation_csv.parent.mkdir(parents=True, exist_ok=True)
    validation.predictions.to_csv(validation_csv, index=False)
    logger.info("Previsões walk-forward salvas: %s", validation_csv)

    save_prediction_chart(validation.predictions, chart_path, validation.mae)
    logger.info("Gráfico previsto vs. realizado salvo: %s", chart_path)

    logger.info("Treinando modelo final e gerando distribuição probabilística")
    prediction_date, central, distribution = fit_final_and_distribution(
        daily,
        n_estimators=int(cfg["n_estimators"]),
        random_state=int(cfg["random_state"]),
    )

    logger.info("Dia previsto: %s", prediction_date.date())
    logger.info("Previsão central de Tmax: %.2f °C", central)
    logger.info("MAE walk-forward final: %.2f °C", validation.mae)

    for temperature, probability in sorted(distribution.items()):
        logger.info("P(Tmax = %d °C) = %.2f%%", temperature, probability * 100.0)

    if market_path.exists():
        market = pd.read_csv(market_path)
        scored = score_market(distribution, market)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        logger.info("Comparação com mercado salva: %s", output_path)
    else:
        logger.warning("market_odds.csv não encontrado; etapa de mercado ignorada")

    logger.info("Execução concluída com sucesso")


if __name__ == "__main__":
    main()
