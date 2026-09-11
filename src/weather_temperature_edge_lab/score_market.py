from __future__ import annotations

import numpy as np
import pandas as pd


def score_market(distribution: dict[int, float], market_df: pd.DataFrame) -> pd.DataFrame:
    required = {"temperature_c", "market_probability"}
    missing = required.difference(market_df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em market_odds.csv: {sorted(missing)}")

    out = market_df.copy()
    out["temperature_c"] = out["temperature_c"].astype(int)
    out["model_probability"] = out["temperature_c"].map(distribution).fillna(0.0)
    out["edge_pp"] = 100.0 * (out["model_probability"] - out["market_probability"])
    out["expected_return_pct"] = np.where(
        out["market_probability"] > 0,
        100.0 * (out["model_probability"] / out["market_probability"] - 1.0),
        np.nan,
    )
    return out.sort_values(["edge_pp", "model_probability"], ascending=[False, False]).reset_index(drop=True)
