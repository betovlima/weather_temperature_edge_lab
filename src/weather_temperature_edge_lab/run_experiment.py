from __future__ import annotations
import json, logging
from pathlib import Path
import pandas as pd
from .build_features import build_daily_features
from .logging_utils import configure_logging
from .plot_results import save_prediction_chart
from .score_market import score_market
from .train_model import fit_final_and_distribution, walk_forward_validate
ROOT=Path(__file__).resolve().parents[2]; logger=logging.getLogger(__name__)
def load_config(): return json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
def main():
    configure_logging(); cfg=load_config(); weather_path=ROOT/cfg["weather_csv"]; market_path=ROOT/cfg["market_csv"]; output_path=ROOT/cfg["output_csv"]
    chart_path=ROOT/"outputs"/"predicted_vs_actual.png"; validation_csv=ROOT/"outputs"/"walk_forward_predictions.csv"
    logger.info("Weather Temperature Edge Lab v0.2.0")
    logger.info("Station timezone: %s | cutoff: %02d:00 local",cfg["station_timezone"],int(cfg["cutoff_hour"]))
    if not weather_path.exists(): raise FileNotFoundError(f"Arquivo não encontrado: {weather_path}")
    hourly=pd.read_csv(weather_path); logger.info("Registros horários carregados: %d",len(hourly))
    daily=build_daily_features(hourly,cutoff_hour=int(cfg["cutoff_hour"]),station_timezone=cfg["station_timezone"])
    if len(daily)<=int(cfg["min_train_days"]): raise ValueError(f"Dias válidos insuficientes: {len(daily)}")
    validation=walk_forward_validate(daily,min_train_days=int(cfg["min_train_days"]),n_estimators=int(cfg["n_estimators"]),random_state=int(cfg["random_state"]),progress_every=int(cfg.get("progress_every",25)))
    validation_csv.parent.mkdir(parents=True,exist_ok=True); validation.predictions.to_csv(validation_csv,index=False)
    logger.info("Auditoria temporal incluída em: %s",validation_csv)
    save_prediction_chart(validation.predictions,chart_path,validation.mae)
    prediction_date,central,distribution=fit_final_and_distribution(daily,n_estimators=int(cfg["n_estimators"]),random_state=int(cfg["random_state"]))
    logger.info("Dia previsto: %s | Tmax central: %.2f °C | MAE: %.2f °C",prediction_date.date(),central,validation.mae)
    for temperature,probability in sorted(distribution.items()): logger.info("P(Tmax = %d °C) = %.2f%%",temperature,probability*100)
    if market_path.exists():
        market=pd.read_csv(market_path); scored=score_market(distribution,market); output_path.parent.mkdir(parents=True,exist_ok=True); scored.to_csv(output_path,index=False)
        logger.warning("market_edge.csv é experimental; valide as odds antes de interpretar edge")
    logger.info("Execução concluída com sucesso")
if __name__=="__main__": main()
