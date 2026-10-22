import yfinance as yf
import pandas as pd

print("Downloading sample market data...")
df = yf.download("AAPL", start="2025-01-01", end="2025-02-01")

print("\nData successfully fetched:")
print(df.head())