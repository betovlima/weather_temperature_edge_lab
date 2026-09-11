from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    mae: float


def feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in df.columns if c not in {"date", "target_tmax_c"}]
    empty = [c for c in candidates if df[c].notna().sum() == 0]
    if empty:
        logger.warning("Features sem nenhum valor observado serão removidas: %s", ", ".join(empty))
    return [c for c in candidates if c not in empty]


def _fit_model(
    train_df: pd.DataFrame,
    columns: list[str],
    n_estimators: int,
    random_state: int,
):
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(train_df[columns])
    y = train_df["target_tmax_c"].to_numpy()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    model.fit(X, y)
    return imputer, model


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:02d}m {seconds:02d}s"


def walk_forward_validate(
    daily_df: pd.DataFrame,
    min_train_days: int = 120,
    n_estimators: int = 500,
    random_state: int = 42,
    progress_every: int = 25,
) -> WalkForwardResult:
    columns = feature_columns(daily_df)
    total = max(0, len(daily_df) - min_train_days)
    records: list[dict] = []

    logger.info("Iniciando backtest walk-forward")
    logger.info("Dias de treino inicial: %d", min_train_days)
    logger.info("Previsões a executar: %d", total)
    logger.info("Features efetivas: %d", len(columns))

    started_at = time.perf_counter()

    for step, i in enumerate(range(min_train_days, len(daily_df)), start=1):
        train = daily_df.iloc[:i]
        test = daily_df.iloc[[i]]

        imputer, model = _fit_model(train, columns, n_estimators, random_state)
        pred = float(model.predict(imputer.transform(test[columns]))[0])
        actual = float(test["target_tmax_c"].iloc[0])

        records.append(
            {
                "date": test["date"].iloc[0],
                "actual_tmax_c": actual,
                "predicted_tmax_c": pred,
                "abs_error_c": abs(actual - pred),
            }
        )

        if step == 1 or step % progress_every == 0 or step == total:
            elapsed = time.perf_counter() - started_at
            rate = elapsed / step
            eta = rate * (total - step)
            current_mae = float(np.mean([r["abs_error_c"] for r in records]))
            pct = 100.0 * step / total if total else 100.0
            logger.info(
                "[%d/%d] %5.1f%% | MAE %.2f °C | ETA %s",
                step,
                total,
                pct,
                current_mae,
                _format_eta(eta),
            )

    pred_df = pd.DataFrame(records)
    mae = (
        float(mean_absolute_error(pred_df["actual_tmax_c"], pred_df["predicted_tmax_c"]))
        if not pred_df.empty
        else float("nan")
    )
    logger.info("Backtest concluído | MAE final %.2f °C", mae)
    return WalkForwardResult(predictions=pred_df, mae=mae)


def fit_final_and_distribution(
    daily_df: pd.DataFrame,
    n_estimators: int = 500,
    random_state: int = 42,
):
    if len(daily_df) < 2:
        raise ValueError("São necessários pelo menos dois dias de dados.")

    train = daily_df.iloc[:-1]
    target = daily_df.iloc[[-1]]
    columns = feature_columns(train)
    imputer, model = _fit_model(train, columns, n_estimators, random_state)
    X = imputer.transform(target[columns])

    central = float(model.predict(X)[0])
    tree_predictions = np.array([tree.predict(X)[0] for tree in model.estimators_])
    rounded = np.rint(tree_predictions).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    probabilities = {int(v): float(c / counts.sum()) for v, c in zip(values, counts)}

    return pd.Timestamp(target["date"].iloc[0]), central, probabilities
