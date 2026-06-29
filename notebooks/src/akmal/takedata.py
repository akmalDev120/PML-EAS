import yfinance as yf

df = yf.download(
    "BBRI.JK",
    start="2025-01-01",
    end="2025-12-31",
    interval="1wk"
)

# Hilangkan MultiIndex
df.columns = df.columns.get_level_values(0)

# Simpan
df.to_csv("bbri_raw.csv")