import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_arima():

    # ==========================
    # LOAD DATA
    # ==========================

    df = pd.read_csv("data/bbri_2tahun.csv")

    df = df.iloc[2:].reset_index(drop=True)

    df.rename(columns={"Price": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"])

    df.set_index("Date", inplace=True)

    # ==========================
    # TRAIN TEST SPLIT
    # ==========================

    train_size = int(len(df) * 0.8)

    train = df["Close"][:train_size]
    test = df["Close"][train_size:]

    # ==========================
    # CARI MODEL TERBAIK
    # ==========================

    orders = [
        (0,1,0),
        (1,1,0),
        (0,1,1),
        (1,1,1),
        (2,1,1),
        (1,1,2)
    ]

    results = []

    for order in orders:

        try:

            model = ARIMA(train, order=order)
            model_fit = model.fit()

            results.append({
                "Model": order,
                "AIC": model_fit.aic
            })

        except:
            continue

    results = pd.DataFrame(results)

    best_order = results.sort_values("AIC").iloc[0]["Model"]

    # ==========================
    # MODEL TERBAIK
    # ==========================

    model = ARIMA(train, order=best_order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))

    mae = mean_absolute_error(test, forecast)
    rmse = mean_squared_error(test, forecast) ** 0.5

    # ==========================
    # GRAFIK
    # ==========================

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(train.index, train, label="Train")
    ax.plot(test.index, test, label="Actual")
    ax.plot(test.index, forecast, label="Forecast")

    ax.set_title(f"Forecast ARIMA {best_order}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Penutupan")

    ax.legend()

    return {
        "figure": fig,
        "mae": mae,
        "rmse": rmse,
        "best_order": best_order,
        "aic_table": results.sort_values("AIC"),
        "summary": str(model_fit.summary())
    }


if __name__ == "__main__":

    hasil = run_arima()

    print("Model Terbaik :", hasil["best_order"])
    print("MAE :", hasil["mae"])
    print("RMSE :", hasil["rmse"])

    print(hasil["aic_table"])

    hasil["figure"].show()