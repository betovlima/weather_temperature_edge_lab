from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_prediction_chart(predictions: pd.DataFrame, output_path: Path, mae: float) -> None:
    if predictions.empty:
        return

    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["date"], df["actual_tmax_c"], label="Realizada", linewidth=2)
    ax.plot(df["date"], df["predicted_tmax_c"], label="Prevista", linewidth=2)
    ax.set_title(f"Temperatura máxima prevista vs. realizada — MAE {mae:.2f} °C")
    ax.set_xlabel("Data")
    ax.set_ylabel("Temperatura máxima (°C)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
