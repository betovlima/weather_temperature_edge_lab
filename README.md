# Weather Temperature Edge Lab

Probabilistic daily maximum temperature forecasting and backtesting using historical weather data, walk-forward validation, and market edge analysis.

## Setup

```powershell
poetry config virtualenvs.in-project true
poetry install
```

## Download weather history

```powershell
poetry run weather-download --station ZUUU --start 2021-01-01 --end 2026-09-11
```

## Run the backtest

```powershell
poetry run weather-backtest
```

The walk-forward backtest reports progress, cumulative MAE and ETA while running.

## Outputs

- `data/weather_hourly.csv`
- `outputs/walk_forward_predictions.csv`
- `outputs/predicted_vs_actual.png`
- `outputs/market_edge.csv`

## Notes

Market probabilities are not used to train the meteorological model. They are compared only after the independent weather probability distribution is generated.
