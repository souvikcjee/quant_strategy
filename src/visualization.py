import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm

def plot_kalman_diagnostics(
    df,
    date_col="Date",
    spread_col="kalman_spread",
    zscore_col="zscore",
    entry_z=1.6,
    acf_lags=40,
    hist_bins_spread=50,
    hist_bins_z=200,
    sigma_xlim=10
):
    """
    Plot full Kalman + Z-score diagnostics:
      1) Kalman spread time series
      2) Kalman spread distribution
      3) ACF of Kalman spread
      4) Z-score distribution with Gaussian fit

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    spread_col : str
    zscore_col : str
    entry_z : float
    acf_lags : int
    hist_bins_spread : int
    hist_bins_z : int
    sigma_xlim : float
    """

    # ===============================
    # 1) Kalman spread time series
    # ===============================
    plt.figure(figsize=(14,5))
    plt.plot(df[date_col], df[spread_col], label="Kalman Spread", alpha=0.8)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Kalman Spread Time Series")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ===============================
    # 2) Spread distribution
    # ===============================
    plt.figure(figsize=(7,5))
    plt.hist(
        df[spread_col].dropna(),
        bins=hist_bins_spread,
        density=True,
        alpha=0.7
    )
    plt.title("Distribution of Kalman Spread")
    plt.xlabel("Spread")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.show()

    # ===============================
    # 3) ACF of spread
    # ===============================
    plt.figure(figsize=(10,4))
    plot_acf(
        df[spread_col].dropna(),
        lags=acf_lags,
        zero=False
    )
    plt.title("ACF of Kalman Spread")
    plt.grid(alpha=0.3)
    plt.show()

    # ===============================
    # 4) Z-score Gaussian fit
    # ===============================
    z = df[zscore_col].dropna().values

    mu_hat, sigma_hat = norm.fit(z)

    x = np.linspace(
        mu_hat - sigma_xlim * sigma_hat,
        mu_hat + sigma_xlim * sigma_hat,
        2000
    )
    pdf = norm.pdf(x, mu_hat, sigma_hat)

    plt.figure(figsize=(7,5))

    plt.hist(
        z,
        bins=hist_bins_z,
        density=True,
        alpha=0.7,
        label="Empirical"
    )

    plt.plot(
        x,
        pdf,
        "k-",
        linewidth=2,
        label="Gaussian fit"
    )

    plt.axvline(entry_z, color="red", linestyle="--", label="Entry Z")
    plt.axvline(-entry_z, color="red", linestyle="--")

    plt.axvline(mu_hat, color="black", linewidth=1)
    plt.axvline(mu_hat + sigma_hat, color="gray", linestyle="--", linewidth=1)
    plt.axvline(mu_hat - sigma_hat, color="gray", linestyle="--", linewidth=1)

    textstr = (
        "Gaussian fit\n"
        f"$\\mu$ = {mu_hat:.3f}\n"
        f"$\\sigma$ = {sigma_hat:.3f}"
    )

    plt.text(
        0.98, 0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.xlim(
        mu_hat - sigma_xlim * sigma_hat,
        mu_hat + sigma_xlim * sigma_hat
    )

    plt.title("Z-score Distribution with Gaussian Fit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ===============================
    # Console diagnostics (important)
    # ===============================
    print("Gaussian Z-score diagnostics")
    print("----------------------------")
    print(f"Mean (μ)   : {mu_hat:.4f}")
    print(f"Std (σ)    : {sigma_hat:.4f}")
    print(f"Entry Z    : ±{entry_z}")
    print(f"Effective Entry (σ-scaled): ±{entry_z * sigma_hat:.2f}")
import matplotlib.pyplot as plt

def plot_equity_and_drawdown(
    df,
    date_col="Date",
    cumret_col="cum_return",
    title="Equity Curve and Drawdown"
):
    """
    Plot cumulative returns and drawdown together.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    cumret_col : str
    title : str
    """

    cum = df[cumret_col]
    peak = cum.cummax()
    drawdown = cum - peak

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14,8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # ===============================
    # Equity curve
    # ===============================
    ax1.plot(df[date_col], cum, linewidth=2)
    ax1.set_title(title)
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(alpha=0.3)

    # ===============================
    # Drawdown
    # ===============================
    ax2.plot(df[date_col], drawdown, color="red")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_trade_and_performance_diagnostics(
    df,
    date_col="Date",
    ret_col="strategy_return",
    trade_id_col="trade_id",
):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    df = df.copy()

    # -------------------------------------------------
    # Hard validation (fail early, fail clearly)
    # -------------------------------------------------
    if ret_col not in df.columns:
        raise ValueError(
            f"{ret_col} not found. Run run_backtest() before plotting."
        )

    if "position" not in df.columns:
        raise ValueError(
            "position column not found. Cannot infer trades."
        )

    # -------------------------------------------------
    # CREATE trade_id AS A COLUMN (CRITICAL FIX)
    # -------------------------------------------------
    if trade_id_col not in df.columns:
        df[trade_id_col] = (df["position"].diff() != 0).cumsum()

    # -------------------------------------------------
    # 1) Monthly returns heatmap
    # -------------------------------------------------
    monthly = (
        df.set_index(date_col)[ret_col]
        .resample("M")
        .sum()
    )

    monthly_df = monthly.to_frame("ret")
    monthly_df["Year"] = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month

    pivot = monthly_df.pivot(
        index="Year",
        columns="Month",
        values="ret"
    )

    plt.figure(figsize=(14,6))
    plt.imshow(pivot, aspect="auto", cmap="RdYlGn")
    plt.colorbar(label="Monthly Return")
    plt.title("Monthly Returns Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.show()

    # -------------------------------------------------
    # 2) Trade return distribution  ✅ NOW SAFE
    # -------------------------------------------------
    trade_pnl = (
        df.groupby(trade_id_col)[ret_col]
        .sum()
    )
    plt.figure(figsize=(7,5))
    plt.hist(trade_pnl, bins=50, alpha=0.7)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Trade Return Distribution")
    plt.xlabel("Trade Return")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()

    # -------------------------------------------------
    # 3) MAE / MFE
    # -------------------------------------------------
    mae, mfe = [], []

    for _, g in df.groupby(trade_id_col):
        if len(g) < 2:
            continue

        pnl_path = g[ret_col].cumsum()
        mae.append(pnl_path.min())
        mfe.append(pnl_path.max())

    mae = np.array(mae)
    mfe = np.array(mfe)

    plt.figure(figsize=(7,6))
    plt.scatter(mae, mfe, alpha=0.6)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.xlabel("MAE")
    plt.ylabel("MFE")
    plt.title("MAE vs MFE per Trade")
    plt.grid(alpha=0.3)
    plt.show()

    print("Trade diagnostics")
    print("-----------------")
    print(f"Trades: {len(trade_pnl)}")
    print(f"Mean MAE: {np.mean(mae):.4f}")
    print(f"Mean MFE: {np.mean(mfe):.4f}")
