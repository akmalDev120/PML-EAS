import yfinance as yf
import pandas as pd

# Download data
df = yf.download(
    "BBRI.JK",
    start="2022-01-01",
    end="2025-12-31"
)

# Informasi awal
print(df.info())

# Cek missing value
print(df.isnull().sum())

# Hapus missing value
df = df.dropna()

# Hapus duplikat
df = df.drop_duplicates()

# Konversi tanggal
df.index = pd.to_datetime(df.index)

# Ambil kolom Close saja
df = df[['Close']]

# Simpan hasil cleaning
df.to_csv("bbri_clean.csv")

print("Data Cleaning Selesai")
print(df.head())