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

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("data/bbri_2tahun.csv")

# hapus baris ticker
df = df.iloc[2:].reset_index(drop=True)

# ubah nama kolom
df.rename(columns={"Price": "Date"}, inplace=True)

# ubah tipe data
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"])

# jadikan index
df.set_index("Date", inplace=True)

print(df.head())

# ==========================
# TRAIN TEST SPLIT
# ==========================

train_size = int(len(df) * 0.8)

train = df["Close"][:train_size]
test = df["Close"][train_size:]

print("Jumlah Train :", len(train))
print("Jumlah Test  :", len(test))

# ==========================
# MEMBANDINGKAN BEBERAPA MODEL ARIMA
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

print("\nPerbandingan Model ARIMA")
print(results.sort_values("AIC"))

# memilih model terbaik
best_order = results.sort_values("AIC").iloc[0]["Model"]

print("\nModel Terbaik :", best_order)

# ==========================
# MEMBANGUN MODEL TERBAIK
# ==========================

model = ARIMA(train, order=best_order)
model_fit = model.fit()

print(model_fit.summary())

# ==========================
# FORECAST
# ==========================

forecast = model_fit.forecast(steps=len(test))

# ==========================
# EVALUASI
# ==========================

mae = mean_absolute_error(test, forecast)
rmse = mean_squared_error(test, forecast) ** 0.5

print("MAE :", mae)
print("RMSE:", rmse)

# ==========================
# VISUALISASI
# ==========================

plt.figure(figsize=(12,6))

plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Forecast")

plt.title(f"Forecast ARIMA {best_order}")

plt.xlabel("Tanggal")
plt.ylabel("Harga Penutupan")

plt.legend()

plt.show()