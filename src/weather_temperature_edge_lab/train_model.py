from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)

AUDIT_COLUMNS = {
    "date", "target_tmax_c", "feature_cutoff_timestamp",
    "last_observation_used", "actual_tmax_timestamp",
}

@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    mae: float


def feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in df.columns if c not in AUDIT_COLUMNS]
    empty = [c for c in candidates if df[c].notna().sum() == 0]
    if empty:
        logger.warning("Features sem nenhum valor observado serão removidas: %s", ", ".join(empty))
    return [c for c in candidates if c not in empty]


def _fit_model(train_df, columns, n_estimators, random_state):
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(train_df[columns])
    y = train_df["target_tmax_c"].to_numpy()
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state,
        n_jobs=-1, min_samples_leaf=2, max_features="sqrt")
    model.fit(X, y)
    return imputer, model


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds)); minutes, seconds = divmod(seconds, 60); hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:02d}m {seconds:02d}s"


def walk_forward_validate(daily_df, min_train_days=120, n_estimators=500, random_state=42, progress_every=25):
    columns = feature_columns(daily_df)
    total = max(0, len(daily_df) - min_train_days); records = []
    logger.info("Iniciando backtest walk-forward | treino inicial=%d | previsões=%d | features=%d", min_train_days, total, len(columns))
    started_at = time.perf_counter()
    for step, i in enumerate(range(min_train_days, len(daily_df)), start=1):
        train = daily_df.iloc[:i]; test = daily_df.iloc[[i]]
        imputer, model = _fit_model(train, columns, n_estimators, random_state)
        pred = float(model.predict(imputer.transform(test[columns]))[0]); actual = float(test["target_tmax_c"].iloc[0])
        records.append({
            "date": test["date"].iloc[0],
            "feature_cutoff_timestamp": test["feature_cutoff_timestamp"].iloc[0],
            "last_observation_used": test["last_observation_used"].iloc[0],
            "actual_tmax_timestamp": test["actual_tmax_timestamp"].iloc[0],
            "actual_tmax_c": actual, "predicted_tmax_c": pred, "abs_error_c": abs(actual-pred),
        })
        if step == 1 or step % progress_every == 0 or step == total:
            elapsed = time.perf_counter()-started_at; eta=(elapsed/step)*(total-step)
            mae=float(np.mean([r["abs_error_c"] for r in records])); pct=100*step/total if total else 100
            logger.info("[%d/%d] %5.1f%% | MAE %.2f °C | ETA %s", step,total,pct,mae,_format_eta(eta))
    pred_df=pd.DataFrame(records)
    mae=float(mean_absolute_error(pred_df["actual_tmax_c"],pred_df["predicted_tmax_c"])) if not pred_df.empty else float("nan")
    logger.info("Backtest concluído | MAE final %.2f °C",mae)
    return WalkForwardResult(predictions=pred_df,mae=mae)


def fit_final_and_distribution(daily_df,n_estimators=500,random_state=42):
    if len(daily_df)<2: raise ValueError("São necessários pelo menos dois dias de dados.")
    train=daily_df.iloc[:-1]; target=daily_df.iloc[[-1]]; columns=feature_columns(train)
    imputer,model=_fit_model(train,columns,n_estimators,random_state); X=imputer.transform(target[columns])
    central=float(model.predict(X)[0]); tree_predictions=np.array([tree.predict(X)[0] for tree in model.estimators_])
    rounded=np.rint(tree_predictions).astype(int); values,counts=np.unique(rounded,return_counts=True)
    probabilities={int(v):float(c/counts.sum()) for v,c in zip(values,counts)}
    return pd.Timestamp(target["date"].iloc[0]),central,probabilities
