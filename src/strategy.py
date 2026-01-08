# src/strategy.py
import numpy as np
import pandas as pd

def build_strategy_kf(
    df: pd.DataFrame,
    half_life: float,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> pd.DataFrame:

    z_window = int(2 * round(half_life))

    # Rolling stats
    df["spread_mean"] = (
        df["kalman_spread"].rolling(z_window).mean().shift(1)
    )
    df["spread_std"] = (
        df["kalman_spread"].rolling(z_window).std().shift(1)
    )

    df["zscore"] = (
        df["kalman_spread"] - df["spread_mean"]
    ) / df["spread_std"]

    # Volatility-aware entry
    df["adj_z"] = df["zscore"] / df["spread_std"]

    df = df.dropna(subset=["adj_z"]).reset_index(drop=True)

    df["position"] = 0
    df["holding_days"] = 0

    position = 0
    days = 0

    for i in range(len(df)):
        z = df.loc[i, "zscore"]

        if position == 0:
            if z < -entry_z:
                position = 1
                days = 0
            elif z > entry_z:
                position = -1
                days = 0
        else:
            days += 1
            if abs(z) < exit_z or days > 2 * half_life:
                position = 0
                days = 0

        df.loc[i, "position"] = position
        df.loc[i, "holding_days"] = days

    return df
