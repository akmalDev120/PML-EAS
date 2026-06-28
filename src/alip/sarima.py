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

train_size = int(len(df)*0.8)

train = df["Close"][:train_size]
test = df["Close"][train_size:]

model = SARIMAX(
    train,
    order=(1,1,2),
    seasonal_order=(1,1,1,5)
)

model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(
    steps=len(test)
)

mae = mean_absolute_error(
    test,
    forecast
)

rmse = mean_squared_error(
    test,
    forecast
) ** 0.5

print("MAE :",mae)
print("RMSE:",rmse)

plt.figure(figsize=(12,6))

plt.plot(train.index,train,label="Train")

plt.plot(test.index,test,label="Actual")

plt.plot(test.index,forecast,label="Forecast")

plt.title("Forecast SARIMA")

plt.legend()

plt.show()

