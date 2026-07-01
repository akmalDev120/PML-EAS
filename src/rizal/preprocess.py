import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Membaca dataset dengan header bertingkat
df = pd.read_csv("data/bbri_2tahun.csv", header=[0, 1])

# Menghapus baris pertama yang berisi "Date"
df = df.iloc[1:].reset_index(drop=True)

# Mengambil kolom tanggal
df["Date"] = pd.to_datetime(df[("Price", "Ticker")])

# Mengambil kolom Close
df["Close"] = df[("Close", "BBRI.JK")].astype(float)

# Menyisakan kolom yang diperlukan
df = df[["Date", "Close"]]

# Normalisasi
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["Close"]])

print(df.head())
print("\nJumlah data:", len(df))
print("\nPreprocessing selesai.")