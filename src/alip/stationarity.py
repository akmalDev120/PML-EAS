import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def run_stationarity():
    # membaca data
    df = pd.read_csv("data/bbri_2tahun.csv")

    # hapus baris ticker
    df = df.iloc[2:].reset_index(drop=True)

    # ubah nama kolom
    df.rename(columns={"Price": "Date"}, inplace=True)

    # ubah tipe data
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"])

    # jadikan tanggal sebagai index
    df.set_index("Date", inplace=True)

    # =========================
    # ADF TEST SEBELUM DIFFERENCING
    # =========================
    adf_before = adfuller(df["Close"])

    # =========================
    # DIFFERENCING
    # =========================
    df["Close_diff"] = df["Close"].diff()
    df = df.dropna()

    # =========================
    # ADF TEST SETELAH DIFFERENCING
    # =========================
    adf_after = adfuller(df["Close_diff"])

    # =========================
    # GRAFIK ACF
    # =========================
    fig_acf, ax1 = plt.subplots(figsize=(10,5))
    plot_acf(df["Close_diff"], lags=30, ax=ax1)

    # =========================
    # GRAFIK PACF
    # =========================
    fig_pacf, ax2 = plt.subplots(figsize=(10,5))
    plot_pacf(df["Close_diff"], lags=30, method="ywm", ax=ax2)

    return {
        "before": adf_before,
        "after": adf_after,
        "acf": fig_acf,
        "pacf": fig_pacf
    }


if __name__ == "__main__":

    hasil = run_stationarity()

    print("===== ADF TEST =====")
    print("ADF Statistic :", hasil["before"][0])
    print("p-value :", hasil["before"][1])

    print("\n===== ADF SETELAH DIFFERENCING =====")
    print("ADF Statistic :", hasil["after"][0])
    print("p-value :", hasil["after"][1])

    hasil["acf"].show()
    hasil["pacf"].show()