# src/stats.py
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

def adf_test(series):
    result = adfuller(series, autolag="AIC")
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "critical_values": result[4],
    }

def estimate_half_life(spread):
    spread_lag = spread.shift(1)
    spread_delta = spread - spread_lag

    spread_lag = spread_lag.iloc[1:]
    spread_delta = spread_delta.iloc[1:]

    X = sm.add_constant(spread_lag)
    model = sm.OLS(spread_delta, X).fit()

    beta = model.params[1]
    half_life = np.log(2) / (-beta)

    return half_life

def kalman_filter(alpha: np.ndarray, state_cov, obs_cov) -> np.ndarray:
    """
    1-state Kalman filter for equilibrium estimation
    """
    kf = KalmanFilter(
        k_endog=1,
        k_states=1,
        initialization="approximate_diffuse"
    )

    kf.design = np.array([[1.0]])
    kf.transition = np.array([[1.0]])
    kf.selection = np.array([[1.0]])

    # Conservative, non-optimized noise
    kf.state_cov = np.array([[state_cov]])
    kf.obs_cov   = np.array([[obs_cov]])

    kf.bind(alpha.astype(float))
    res = kf.filter()

    return res.filtered_state[0, :]  # shape (T,)
