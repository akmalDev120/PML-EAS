import yfinance as yf

df = yf.download("BBRI.JK", period="2y")
df.to_csv("bbri_2tahun.csv")