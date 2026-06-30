import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def run_sarima():

    # ==========================
    # LOAD DATA
    # ==========================

    df = pd.read_csv("data/bbri_2tahun.csv")

    df = df.iloc[2:].reset_index(drop=True)

    df.rename(columns={"Price":"Date"}, inplace=True)

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
    # MODEL SARIMA
    # ==========================

    model = SARIMAX(
        train,
        order=(1,1,2),
        seasonal_order=(1,1,1,5)
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(
        steps=len(test)
    )

    # ==========================
    # EVALUASI
    # ==========================

    mae = mean_absolute_error(test, forecast)

    rmse = mean_squared_error(
        test,
        forecast
    ) ** 0.5

    # ==========================
    # VISUALISASI
    # ==========================

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(train.index, train, label="Train")

    ax.plot(test.index, test, label="Actual")

    ax.plot(test.index, forecast, label="Forecast")

    ax.set_title("Forecast SARIMA")

    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Penutupan")

    ax.legend()

    return {
        "figure": fig,
        "mae": mae,
        "rmse": rmse,
        "summary": model_fit.summary().as_text()
    }


if __name__ == "__main__":

    hasil = run_sarima()

    print(hasil["summary"])

    print("MAE :", hasil["mae"])
    print("RMSE :", hasil["rmse"])

    hasil["figure"].show()