import pandas as pd
from prophet import Prophet

# ===========================
# Membaca dataset
# ===========================
df = pd.read_csv("data/bbri_2tahun.csv", header=[0,1])

# Menghapus baris pertama yang kosong
df = df.iloc[1:].reset_index(drop=True)

# Membuat dataframe untuk Prophet
data = pd.DataFrame({
    "ds": pd.to_datetime(df[("Price", "Ticker")]),
    "y": df[("Close", "BBRI.JK")].astype(float)
})

print("===== DATASET =====")
print(data.head())

print("\nJumlah data:", len(data))

# ===========================
# Training Model Prophet
# ===========================
print("\nMelatih model Prophet...")

model = Prophet()

model.fit(data)

print("\n✅ Model Prophet berhasil dilatih.")

# ===========================
# Prediksi 30 Hari ke Depan
# ===========================
future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

print("\n===== HASIL PREDIKSI 5 HARI TERAKHIR =====")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# ===========================
# Simpan Hasil Prediksi
# ===========================
forecast.to_csv("data/hasil_prophet.csv", index=False)

print("\n✅ Hasil prediksi berhasil disimpan.")
print("Lokasi file : data/hasil_prophet.csv")

# ===========================
# Menampilkan Grafik
# ===========================
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)

print("\nGrafik prediksi berhasil dibuat.")