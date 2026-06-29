import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Tanggal hari ini
today = datetime.today()

# Ambil data 2 bulan terakhir
start_date = today - relativedelta(months=2)

# Download data historis
df = yf.download(
    "BBRI.JK",
    start=start_date.strftime("%Y-%m-%d"),
    end=today.strftime("%Y-%m-%d")
)

# Informasi awal
print(df.info())

# Cek missing value
print("\nMissing Value:")
print(df.isnull().sum())

# Data Cleaning
df = df.dropna()
df = df.drop_duplicates()

# Pastikan index berupa datetime
df.index = pd.to_datetime(df.index)

# Ambil kolom Close
df = df[['Close']]

# Simpan hasil
df.to_csv("bbri_clean.csv")

print("\nData Cleaning Selesai")
print(df.head())
print(f"\nJumlah data: {len(df)}")