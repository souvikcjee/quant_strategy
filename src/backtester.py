# src/backtester.py
import numpy as np

# ======================================================
# Backtest execution + PnL accounting
# ======================================================
def run_backtest(
    df,
    transaction_cost=0.0002,
    target_vol=0.01,
    vol_window=20,
    max_leverage=3.0,
):
    """
    Apply execution model, volatility targeting, and transaction costs.

    Assumes df already contains:
      - position
      - log_return
    """

    df = df.copy()

    # ------------------------------
    # Volatility targeting
    # ------------------------------
    df["ret_vol"] = (
        df["log_return"]
        .rolling(vol_window)
        .std()
        .shift(1)
    )

    # Avoid division by zero
    df["vol_scaler"] = target_vol / df["ret_vol"]
    df["vol_scaler"] = df["vol_scaler"].replace([np.inf, -np.inf], np.nan)
    df["vol_scaler"] = df["vol_scaler"].clip(0, max_leverage)

    # ------------------------------
    # Strategy returns (T+1 execution)
    # ------------------------------
    df["strategy_return"] = (
        df["position"].shift(1)
        * df["vol_scaler"].shift(1)
        * df["log_return"]
    )

    # ------------------------------
    # Transaction costs
    # ------------------------------
    trades = df["position"].diff().abs()
    df["strategy_return"] -= trades * transaction_cost
    
    df = df.dropna(subset=["strategy_return"]).reset_index(drop=True)
    return df


# ======================================================
# Performance metrics
# ======================================================
def performance_metrics(df, ann_factor=252):
    """
    Compute standard performance metrics.
    """

    r = df["strategy_return"]

    annual_return = r.mean() * ann_factor
    annual_vol = r.std() * np.sqrt(ann_factor)
    sharpe = (
        annual_return / annual_vol
        if annual_vol > 0 else np.nan
    )

    # ------------------------------
    # Drawdown
    # ------------------------------
    cum = r.cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd = drawdown.min()

    calmar = (
        annual_return / abs(max_dd)
        if max_dd < 0 else np.nan
    )

    # ------------------------------
    # Trade-level stats
    # ------------------------------
    trade_id = (df["position"].diff() != 0).cumsum()
    trade_pnl = (
        df.groupby(trade_id)["strategy_return"]
        .sum()
    )

    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl <= 0]

    return {
        "Annual Return": annual_return,
        "Annual Vol": annual_vol,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_dd,
        "Win Rate": len(wins) / len(trade_pnl) if len(trade_pnl) > 0 else np.nan,
        "Total Trades": len(trade_pnl),
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
    }
