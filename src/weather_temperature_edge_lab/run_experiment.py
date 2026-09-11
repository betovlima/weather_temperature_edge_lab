from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .build_features import build_daily_features
from .logging_utils import configure_logging
from .plot_information_curve import save_information_curve
from .train_model import walk_forward_validate
ROOT=Path(__file__).resolve().parents[2]; logger=logging.getLogger(__name__)
def load_config(): return json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
def _metrics(p):
    e=p["predicted_tmax_c"]-p["actual_tmax_c"]; re=np.rint(p["predicted_tmax_c"])-np.rint(p["actual_tmax_c"])
    return {"mae_c":float(e.abs().mean()),"rmse_c":float(np.sqrt(np.mean(e**2))),"bias_c":float(e.mean()),"exact_bucket_pct":float((re==0).mean()*100),"within_1c_bucket_pct":float((re.abs()<=1).mean()*100)}
def _baseline_metrics(daily,dates):
    f=daily.set_index("date").copy(); f["previous_day_tmax"]=f["target_tmax_c"].shift(1); s=f.loc[pd.to_datetime(dates)]; actual=s["target_tmax_c"]; rows=[]
    for name,pred in {"previous_day_tmax":s["previous_day_tmax"],"max_observed_at_cutoff":s["temp_max_so_far"]}.items():
        valid=pred.notna()&actual.notna(); e=pred[valid]-actual[valid]; re=np.rint(pred[valid])-np.rint(actual[valid])
        rows.append({"baseline":name,"mae_c":float(e.abs().mean()),"rmse_c":float(np.sqrt(np.mean(e**2))),"bias_c":float(e.mean()),"exact_bucket_pct":float((re==0).mean()*100),"within_1c_bucket_pct":float((re.abs()<=1).mean()*100)})
    return rows
def main():
    configure_logging(); cfg=load_config(); weather_path=ROOT/cfg["weather_csv"]
    if not weather_path.exists(): raise FileNotFoundError(f"Arquivo não encontrado: {weather_path}")
    hourly=pd.read_csv(weather_path); cutoffs=[int(h) for h in cfg.get("cutoff_hours",[cfg["cutoff_hour"]])]; out=ROOT/"outputs"; out.mkdir(parents=True,exist_ok=True); summaries=[]; baseline_rows=[]
    logger.info("Weather Temperature Edge Lab v0.3.0 | cutoffs locais: %s",", ".join(f"{h:02d}:00" for h in cutoffs))
    for pos,cutoff in enumerate(cutoffs,1):
        logger.info("=== Cutoff %02d:00 (%d/%d) ===",cutoff,pos,len(cutoffs)); daily=build_daily_features(hourly,cutoff_hour=cutoff,station_timezone=cfg["station_timezone"])
        v=walk_forward_validate(daily,min_train_days=int(cfg["min_train_days"]),n_estimators=int(cfg["n_estimators"]),random_state=int(cfg["random_state"]),progress_every=int(cfg.get("progress_every",25)))
        p=v.predictions.copy(); p.insert(1,"cutoff_hour_local",cutoff); p.to_csv(out/f"walk_forward_predictions_{cutoff:02d}h.csv",index=False); m=_metrics(p); summaries.append({"cutoff_hour_local":cutoff,"n_predictions":len(p),**m})
        for row in _baseline_metrics(daily,p["date"]): baseline_rows.append({"cutoff_hour_local":cutoff,**row})
        logger.info("Cutoff %02dh | MAE %.3f C | exato %.1f%% | ±1C %.1f%%",cutoff,m["mae_c"],m["exact_bucket_pct"],m["within_1c_bucket_pct"])
    summary=pd.DataFrame(summaries).sort_values("cutoff_hour_local"); baselines=pd.DataFrame(baseline_rows).sort_values(["cutoff_hour_local","baseline"]); summary.to_csv(out/"multi_cutoff_summary.csv",index=False); baselines.to_csv(out/"baseline_summary.csv",index=False); save_information_curve(summary,out/"multi_cutoff_information_curve.png")
    logger.info("=== Curva de informação ===")
    for r in summary.itertuples(index=False): logger.info("%02dh | MAE %.3f C | exato %.1f%% | ±1C %.1f%%",r.cutoff_hour_local,r.mae_c,r.exact_bucket_pct,r.within_1c_bucket_pct)
    logger.info("Execução v0.3.0 concluída | %s",out)
if __name__=="__main__": main()
