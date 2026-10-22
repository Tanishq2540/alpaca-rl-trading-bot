import argparse

import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


class SmaCross(Strategy):
    fast = 20
    slow = 50

    def init(self):
        close = self.data.Close
        self.sma_fast = self.I(
            lambda x: pd.Series(x).rolling(self.fast).mean(), close, name="SMA fast"
        )
        self.sma_slow = self.I(
            lambda x: pd.Series(x).rolling(self.slow).mean(), close, name="SMA slow"
        )

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.position.close()
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()
            self.sell()


def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise SystemExit(f"No data returned for {ticker} between {start} and {end}")
    return df


def main():
    parser = argparse.ArgumentParser(description="SMA crossover backtest")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--cash", type=float, default=10_000)
    parser.add_argument("--commission", type=float, default=0.002)
    parser.add_argument("--plot", action="store_true", help="open an interactive plot")
    args = parser.parse_args()

    data = fetch_data(args.ticker, args.start, args.end)
    print(f"{args.ticker}: {len(data)} rows from {data.index[0].date()} to {data.index[-1].date()}")

    bt = Backtest(
        data, SmaCross, cash=args.cash, commission=args.commission, finalize_trades=True
    )
    stats = bt.run(fast=args.fast, slow=args.slow)
    print(stats)

    if args.plot:
        bt.plot()


if __name__ == "__main__":
    main()
