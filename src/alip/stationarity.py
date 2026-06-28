import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

print(df.head())

# =========================
# UJI STASIONERITAS (ADF TEST)
# =========================

result = adfuller(df["Close"])

print("===== HASIL ADF TEST =====")
print(f"ADF Statistic : {result[0]}")
print(f"p-value       : {result[1]}")

print("\nCritical Values:")
for key, value in result[4].items():
    print(f"{key} : {value}")

# Interpretasi
if result[1] < 0.05:
    print("\nKesimpulan: Data SUDAH stasioner.")
else:
    print("\nKesimpulan: Data BELUM stasioner.")


# =========================
# DIFFERENCING
# =========================

df["Close_diff"] = df["Close"].diff()

df = df.dropna()

print(df.head())

# =========================
# ADF TEST SETELAH DIFFERENCING
# =========================

result = adfuller(df["Close_diff"])

print("\n===== ADF TEST SETELAH DIFFERENCING =====")
print(f"ADF Statistic : {result[0]}")
print(f"p-value       : {result[1]}")

print("\nCritical Values:")
for key, value in result[4].items():
    print(f"{key} : {value}")

if result[1] < 0.05:
    print("\nKesimpulan: Data SUDAH stasioner.")
else:
    print("\nKesimpulan: Data BELUM stasioner.")

# =========================
# PLOT ACF DAN PACF
# =========================

plt.figure(figsize=(10, 5))
plot_acf(df["Close_diff"], lags=30)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(df["Close_diff"], lags=30, method="ywm")
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()