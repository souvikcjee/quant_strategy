import pandas as pd
import requests
from datetime import datetime, timedelta
import zipfile
import io
import os

def download_nse_fo_bhavcopy(start_date, end_date, out_csv,F_SYMBOL="BHARTIARTL"):
    all_data = []

    d = start_date
    while d <= end_date:
        date_str = d.strftime("%d%b%Y").upper()
        url = f"https://archives.nseindia.com/content/historical/DERIVATIVES/{d.year}/{d.strftime('%b').upper()}/fo{date_str}bhav.csv.zip"

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                csv_name = z.namelist()[0]
                df = pd.read_csv(z.open(csv_name))

                df = df[
                    (df["SYMBOL"] == F_SYMBOL) &
                    (df["INSTRUMENT"] == "FUTSTK")
                ]

                if not df.empty:
                    df["DATE"] = d
                    all_data.append(df)

        except Exception:
            pass

        d += timedelta(days=1)

    if not all_data:
        raise RuntimeError("No data downloaded")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(final_df)} rows")


start = datetime.today() - timedelta(days=365*10)
end = datetime.today()

download_nse_fo_bhavcopy(start, end, "data_futures_raw.csv")
