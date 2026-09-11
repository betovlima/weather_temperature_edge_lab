from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_information_curve(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary["cutoff_hour_local"], summary["mae_c"], marker="o", label="MAE (C)")
    ax.set_xlabel("Local cutoff hour")
    ax.set_ylabel("MAE (C)")
    ax.set_title("Temperature information curve by local cutoff")
    ax.set_xticks(summary["cutoff_hour_local"])
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
