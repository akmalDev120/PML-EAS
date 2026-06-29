from pathlib import Path
import pandas as pd

# Folder tempat cleaning.py berada
BASE_DIR = Path(__file__).resolve().parent

# Lokasi file
raw_file = BASE_DIR / "bbri_raw.csv"
clean_file = BASE_DIR / "bbri_clean.csv"

# Membaca data
df = pd.read_csv(raw_file)

print("===== DATA AWAL =====")
print(df.head())

# Informasi data
print("\n===== INFO DATA =====")
print(df.info())

# Cek missing value
print("\n===== MISSING VALUE =====")
print(df.isnull().sum())

# Hapus missing value
df = df.dropna()

# Hapus data duplikat
df = df.drop_duplicates()

# Konversi kolom Date menjadi datetime
df["Date"] = pd.to_datetime(df["Date"])

# Pastikan kolom numerik bertipe angka
numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# Urutkan berdasarkan tanggal
df = df.sort_values("Date")

# Simpan data bersih
df.to_csv(clean_file, index=False)

print("\n===== DATA SETELAH CLEANING =====")
print(df.head())

print("\nJumlah data:", len(df))
print("Data bersih berhasil disimpan ke:", clean_file)
print("Jumlah data duplikat:", df.duplicated().sum())