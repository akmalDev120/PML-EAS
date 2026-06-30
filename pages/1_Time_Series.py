import streamlit as st

from src.alip.stationarity import run_stationarity
from src.alip.arima import run_arima
from src.alip.sarima import run_sarima

st.set_page_config(page_title="Time Series Analysis")

st.title("📊 Time Series Analysis")

st.markdown("""
Analisis Time Series digunakan untuk mengetahui pola historis harga saham
**BBRI** menggunakan metode statistik.

Metode yang digunakan:

- 📈 ADF Test
- 📊 ACF
- 📉 PACF
- 🤖 ARIMA
- 🤖 SARIMA
""")

st.divider()

# =========================================
# STATIONARITY
# =========================================

st.info("""
Tahapan pertama adalah menguji apakah data sudah stasioner menggunakan
Augmented Dickey Fuller (ADF Test).
""")

st.caption("""
ACF digunakan untuk menentukan orde MA,
sedangkan PACF digunakan untuk menentukan orde AR.
""")

st.info("""
Model ARIMA dipilih berdasarkan nilai AIC.
Semakin kecil nilai AIC maka model semakin baik.
""")

hasil = run_stationarity()

hasil_arima = run_arima()
hasil_sarima = run_sarima()

st.write(hasil_arima.keys())

st.header("1️⃣ Uji Stasioneritas (ADF Test)")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Status Stasioneritas",
    "✅ Stationary" if hasil["after"][1] < 0.05 else "❌ Non-Stationary"
)

c2.metric(
    "Best ARIMA",
    str(hasil_arima["best_order"])
)

c3.metric(
    "RMSE Terbaik",
    f"{min(hasil_arima['rmse'], hasil_sarima['rmse']):.2f}"
)

st.divider()

st.subheader("Sebelum Differencing")

col1, col2 = st.columns(2)

col1.metric(
    "ADF Statistic",
    f"{hasil['before'][0]:.4f}"
)

col2.metric(
    "p-value",
    f"{hasil['before'][1]:.6f}"
)

if hasil["before"][1] < 0.05:
    st.success("Data sudah stasioner")
else:
    st.warning("Data belum stasioner")

st.subheader("Setelah Differencing")

col1, col2 = st.columns(2)

col1.metric(
    "ADF Statistic",
    f"{hasil['after'][0]:.4f}"
)

col2.metric(
    "p-value",
    f"{hasil['after'][1]:.6f}"
)

if hasil["after"][1] < 0.05:
    st.success("Data sudah stasioner")
else:
    st.warning("Data belum stasioner")

st.subheader("Autocorrelation Function (ACF)")
st.pyplot(hasil["acf"])

st.subheader("Partial Autocorrelation Function (PACF)")
st.pyplot(hasil["pacf"])

st.subheader("📝 Kesimpulan Uji Stasioneritas")

if hasil["after"][1] < 0.05:
    st.success("""
Data telah memenuhi asumsi stasioneritas setelah dilakukan differencing.

Hal ini ditunjukkan oleh nilai **p-value < 0.05**, sehingga data layak digunakan
untuk proses pemodelan menggunakan ARIMA maupun SARIMA.
""")
else:
    st.error("""
Data belum stasioner karena nilai **p-value > 0.05**.
Disarankan melakukan differencing kembali sebelum membangun model.
""")

st.divider()

# =========================================
# ARIMA
# =========================================

st.header("2️⃣ Forecast ARIMA")

st.write("### Perbandingan Model Berdasarkan AIC")

st.dataframe(
    hasil_arima["aic_table"],
    use_container_width=True
)

st.success(f"Model Terbaik : {hasil_arima['best_order']}")

c1, c2 = st.columns(2)

c1.metric(
    "MAE",
    f"{hasil_arima['mae']:.2f}"
)

c2.metric(
    "RMSE",
    f"{hasil_arima['rmse']:.2f}"
)

st.pyplot(hasil_arima["figure"])

with st.expander("📄 Lihat Ringkasan Model ARIMA"):
    st.text(hasil_arima["summary"])
    
if hasil_arima["rmse"] < 500:
    st.success("Model ARIMA memiliki tingkat error yang rendah.")
else:
    st.warning("Model ARIMA masih memiliki error yang cukup tinggi.")

st.subheader("📝 Kesimpulan Model ARIMA")

st.write(f"""
Model terbaik yang diperoleh berdasarkan nilai **AIC** adalah **ARIMA {hasil_arima['best_order']}**.

Hasil evaluasi model menunjukkan:

- MAE : {hasil_arima['mae']:.2f}
- RMSE : {hasil_arima['rmse']:.2f}

Nilai MAE dan RMSE yang relatif kecil menunjukkan bahwa model mampu mengikuti pola data historis dengan cukup baik.
""")

st.divider()

# =========================================
# SARIMA
# =========================================

st.info("""
SARIMA merupakan pengembangan ARIMA yang mempertimbangkan pola musiman (seasonality).
""")

st.header("3️⃣ Forecast SARIMA")

c1, c2 = st.columns(2)

c1.metric(
    "MAE",
    f"{hasil_sarima['mae']:.2f}"
)

c2.metric(
    "RMSE",
    f"{hasil_sarima['rmse']:.2f}"
)

st.pyplot(hasil_sarima["figure"])

st.subheader("📝 Kesimpulan Model SARIMA")

st.write(f"""
Model SARIMA digunakan untuk menangkap pola musiman pada data time series.

Hasil evaluasi model:

- MAE : {hasil_sarima['mae']:.2f}
- RMSE : {hasil_sarima['rmse']:.2f}

Model SARIMA memberikan hasil prediksi yang dapat digunakan sebagai pembanding terhadap model ARIMA.
""")

if hasil_sarima["rmse"] < 500:
    st.success("Model SARIMA memberikan hasil prediksi yang baik.")
else:
    st.warning("Model SARIMA masih perlu dilakukan tuning.")

with st.expander("Lihat Ringkasan Model SARIMA"):
    st.text(hasil_sarima["summary"])

st.caption("""
Dataset : Yahoo Finance (BBRI.JK)

Metode:
• ADF Test
• ARIMA
• SARIMA

Developed for EAS Machine Learning
""")