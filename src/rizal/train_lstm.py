import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===========================
# Membaca Dataset
# ===========================
df = pd.read_csv("data/bbri_2tahun.csv", header=[0,1])

df = df.iloc[1:].reset_index(drop=True)

df["Date"] = pd.to_datetime(df[("Price","Ticker")])
df["Close"] = df[("Close","BBRI.JK")].astype(float)

data = df[["Close"]]

print("===== DATASET =====")
print(data.head())

# ===========================
# Normalisasi Data
# ===========================
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(data)

print("\nJumlah data :", len(scaled))

# ===========================
# Membuat Sequence
# ===========================
time_step = 30

X = []
y = []

for i in range(time_step, len(scaled)):
    X.append(scaled[i-time_step:i,0])
    y.append(scaled[i,0])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

print("Shape X :", X.shape)
print("Shape y :", y.shape)

# ===========================
# Split Train dan Test
# ===========================
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("\nData Train :", len(X_train))
print("Data Test  :", len(X_test))

# ===========================
# Membuat Model LSTM
# ===========================
model = Sequential()

model.add(LSTM(50, activation='tanh', input_shape=(time_step,1)))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

print("\n===== Ringkasan Model =====")
model.summary()

# ===========================
# Training Model
# ===========================
print("\nMelatih model...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test,y_test),
    verbose=1
)

print("\nTraining selesai.")

# ===========================
# Prediksi
# ===========================
prediction = model.predict(X_test)

prediction = scaler.inverse_transform(prediction.reshape(-1,1))
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# ===========================
# Evaluasi
# ===========================
mae = mean_absolute_error(actual,prediction)

rmse = np.sqrt(mean_squared_error(actual,prediction))

mape = np.mean(np.abs((actual-prediction)/actual))*100

print("\n===== HASIL EVALUASI =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# ===========================
# Simpan Model
# ===========================
model.save("data/model_lstm.keras")

print("\nModel berhasil disimpan.")
print("Lokasi : data/model_lstm.keras")

# ===========================
# Grafik
# ===========================
plt.figure(figsize=(12,6))
plt.plot(actual, label="Data Asli")
plt.plot(prediction, label="Prediksi")
plt.title("Prediksi Harga Saham BBRI Menggunakan LSTM")
plt.xlabel("Waktu")
plt.ylabel("Harga")
plt.legend()

plt.savefig("data/grafik_lstm.png")

print("\nGrafik disimpan di data/grafik_lstm.png")

plt.show()