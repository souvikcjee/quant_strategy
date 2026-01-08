# src/data_loader.py
import pandas as pd
import numpy as np

def load_data(csv_path: str,filter_expiry=(1, 2)) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse Trade Date
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    # Parse Expiry Date
    df["Expiry Date"] = pd.to_datetime(
        df["Expiry Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    # Clean CLOSE price
    df["CLOSE"] = (
        df["Close Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("-", np.nan)
        .astype(float)
    )

    # Drop invalid rows
    df = df.dropna(subset=["Date", "Expiry Date", "CLOSE"]).reset_index(drop=True)

    #df["Expiry Date"] = pd.to_datetime(df["Expiry Date"])

    # Days to expiry (still useful)
    df["number_of_days_to_expiry"] = (
        df["Expiry Date"] - df["Date"]
    ).dt.days
    
    # Remove expired contracts
    df = df[df["number_of_days_to_expiry"] >= 0].copy()
    
    # 🔑 Label expiry order within each Date
    df["expiry_rank"] = (
        df.groupby("Date")["Expiry Date"]
          .rank(method="first", ascending=True)
          .astype(int)
    )
    df = (
    df.sort_values(by=["Date", "Expiry Date"])
      .reset_index(drop=True)
    )

    df = df[df["expiry_rank"].isin(filter_expiry)].reset_index(drop=True)
    return df
# src/data_exploration.py
import pandas as pd
import numpy as np

def correlation_analysis(df: pd.DataFrame):
    df = df.copy()

    # -----------------------------
    # Log price & log return
    # -----------------------------
    df["log_price"] = np.log(df["CLOSE"])
    df["log_return"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))

    # Drop first NaN return
    df = df.dropna(subset=["log_return"])

    # -----------------------------
    # Columns to test
    # -----------------------------
    feature_cols = [
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Last Price",
        "Settlement Price",
        "Volume",
        "Open Interest",
        "Change in OI",
        "number_of_days_to_expiry",
    ]

    # Convert to numeric safely
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_cols)

    # -----------------------------
    # Correlation matrices
    # -----------------------------
    corr_log_price = (
        df[feature_cols]
        .corrwith(df["log_price"])
        .sort_values(ascending=False)
    )

    corr_log_return = (
        df[feature_cols]
        .corrwith(df["log_return"])
        .sort_values(ascending=False)
    )

    return corr_log_price, corr_log_return
