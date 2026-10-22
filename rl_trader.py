import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from gymnasium import spaces
from stable_baselines3 import PPO


def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise SystemExit(f"No data returned for {ticker} between {start} and {end}")
    return df


def build_features(df):
    close = df["Close"]
    volume = df["Volume"]
    ret = close.pct_change()
    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    features = pd.DataFrame(
        {
            "Close_return": ret,
            "Close_vs_SMA10": close / sma10 - 1,
            "Close_vs_SMA50": close / sma50 - 1,
            "SMA10_vs_SMA50": sma10 / sma50 - 1,
            "RSI": (rsi - 50) / 50,
            "MACD": macd / close,
            "MACD_hist": macd_hist / close,
            "Volatility": ret.rolling(20).std(),
            "Volume_z": (volume - volume.rolling(20).mean())
            / volume.rolling(20).std(),
        },
        index=df.index,
    )
    features["Volume_z"] = features["Volume_z"].clip(-5, 5)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    dates = features.index
    prices = close.loc[dates].to_numpy(dtype=np.float32)
    return features.to_numpy(dtype=np.float32), prices, dates


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features,
        prices,
        window=20,
        initial_cash=10_000.0,
        fee=0.001,
        trade_penalty=0.0005,
        reward_window=20,
        sharpe_weight=1.0,
        downside_penalty=2.0,
    ):
        super().__init__()
        self.features = features
        self.prices = prices
        self.window = window
        self.initial_cash = initial_cash
        self.fee = fee
        self.trade_penalty = trade_penalty
        self.reward_window = reward_window
        self.sharpe_weight = sharpe_weight
        self.downside_penalty = downside_penalty
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window * features.shape[1],),
            dtype=np.float32,
        )

    def _observation(self):
        start = self.step_idx - self.window
        return self.features[start : self.step_idx].reshape(-1).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = self.window
        self.cash = 1.0
        self.shares = 0.0
        self.trade_count = 0
        self.total_costs = 0.0
        self.recent_returns = deque(maxlen=self.reward_window)
        return self._observation(), {}

    def step(self, action):
        price = float(self.prices[self.step_idx])
        prev_equity = self.cash + self.shares * price

        if action == 1 and self.shares == 0.0:
            notional = self.cash
            penalty = notional * self.trade_penalty
            invested = notional - penalty
            self.shares = invested / (price * (1 + self.fee))
            self.cash = 0.0
            self.total_costs += penalty + invested * self.fee / (1 + self.fee)
            self.trade_count += 1
        elif action == 2 and self.shares > 0.0:
            gross = self.shares * price
            fee_cost = gross * self.fee
            penalty = gross * self.trade_penalty
            self.cash = gross - fee_cost - penalty
            self.shares = 0.0
            self.total_costs += fee_cost + penalty
            self.trade_count += 1

        self.step_idx += 1
        done = self.step_idx >= len(self.prices)
        next_price = float(self.prices[min(self.step_idx, len(self.prices) - 1)])
        equity = self.cash + self.shares * next_price
        step_return = equity / prev_equity - 1.0
        self.recent_returns.append(step_return)
        returns = np.asarray(self.recent_returns)

        sharpe = 0.0
        if len(returns) >= 2:
            std = float(returns.std())
            if std > 1e-8:
                sharpe = float(returns.mean() / std)
        sharpe = float(np.clip(sharpe, -5.0, 5.0))
        downside_dev = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
        reward = self.sharpe_weight * sharpe - self.downside_penalty * downside_dev

        info = {
            "equity": equity,
            "net_worth": equity * self.initial_cash,
            "costs": self.total_costs * self.initial_cash,
            "shares": self.shares,
            "price": next_price,
            "trades": self.trade_count,
            "sharpe": sharpe,
            "downside_dev": downside_dev,
        }
        return self._observation(), reward, done, False, info


def max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min() * 100)


def evaluate(
    model,
    features,
    prices,
    window,
    initial_cash,
    fee,
    trade_penalty,
    reward_window,
    sharpe_weight,
    downside_penalty,
):
    env = TradingEnv(
        features,
        prices,
        window,
        initial_cash,
        fee,
        trade_penalty,
        reward_window,
        sharpe_weight,
        downside_penalty,
    )
    obs, _ = env.reset()
    done = False
    equity_curve = [1.0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        equity_curve.append(info["equity"])
    growth = equity_curve[-1]
    buy_hold_final = initial_cash * prices[-1] / prices[window]
    buy_hold = (prices[-1] / prices[window] - 1) * 100
    return {
        "final_equity": initial_cash * growth,
        "return_pct": (growth - 1) * 100,
        "buy_hold_final": float(buy_hold_final),
        "buy_hold_pct": float(buy_hold),
        "max_drawdown_pct": max_drawdown(np.asarray(equity_curve)),
        "trades": info["trades"],
        "costs": info["costs"],
        "equity_curve": equity_curve,
    }


def main():
    parser = argparse.ArgumentParser(description="PPO trading agent for a single stock")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--eval-start", default="2023-01-01")
    parser.add_argument("--eval-end", default="2025-01-01")
    parser.add_argument("--cash", type=float, default=10_000)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--trade-penalty", type=float, default=0.0005)
    parser.add_argument("--reward-window", type=int, default=20)
    parser.add_argument("--sharpe-weight", type=float, default=1.0)
    parser.add_argument("--downside-penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="ppo_trader")
    parser.add_argument("--weights-out", default="ppo_trader_weights.pth")
    args = parser.parse_args()

    df = fetch_data(args.ticker, args.start, args.end)
    features, prices, dates = build_features(df)

    eval_start_ts = pd.Timestamp(args.eval_start)
    eval_end_ts = pd.Timestamp(args.eval_end)
    if eval_end_ts <= eval_start_ts:
        raise SystemExit("--eval-end must be after --eval-start")
    in_eval = np.asarray((dates >= eval_start_ts) & (dates < eval_end_ts))
    if not in_eval.any():
        raise SystemExit(
            f"No data between {args.eval_start} and {args.eval_end} for {args.ticker}"
        )
    first_eval = int(np.argmax(in_eval))
    last_eval = len(dates) - int(np.argmax(in_eval[::-1]))
    if first_eval < args.window:
        raise SystemExit("Not enough training history before the evaluation period")

    train_features, train_prices = features[:first_eval], prices[:first_eval]
    eval_features = features[first_eval - args.window : last_eval]
    eval_prices = prices[first_eval - args.window : last_eval]

    print(
        f"{args.ticker}: train={len(train_prices)} samples "
        f"({dates[0].date()} to {dates[first_eval - 1].date()}), "
        f"out-of-sample={last_eval - first_eval} samples "
        f"({dates[first_eval].date()} to {dates[last_eval - 1].date()})"
    )

    train_env = TradingEnv(
        train_features,
        train_prices,
        args.window,
        args.cash,
        args.fee,
        args.trade_penalty,
        args.reward_window,
        args.sharpe_weight,
        args.downside_penalty,
    )
    print(
        f"Reward: rolling Sharpe (window={args.reward_window}, "
        f"weight={args.sharpe_weight}) - "
        f"{args.downside_penalty} x downside deviation"
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}.zip")
    torch.save(model.policy.state_dict(), args.weights_out)
    print(f"Saved policy weights to {args.weights_out}")

    results = evaluate(
        model,
        eval_features,
        eval_prices,
        args.window,
        args.cash,
        args.fee,
        args.trade_penalty,
        args.reward_window,
        args.sharpe_weight,
        args.downside_penalty,
    )
    print(
        f"\nOut-of-sample evaluation ({dates[first_eval].date()} "
        f"to {dates[last_eval - 1].date()})"
    )
    print(f"  PPO final net worth:        ${results['final_equity']:,.2f}")
    print(f"  Buy & hold final net worth: ${results['buy_hold_final']:,.2f}")
    print(f"  PPO return:                 {results['return_pct']:.2f}%")
    print(f"  Buy & hold return:          {results['buy_hold_pct']:.2f}%")
    print(f"  Max drawdown:               {results['max_drawdown_pct']:.2f}%")
    print(f"  Trades:                     {results['trades']}")
    print(f"  Transaction costs:          ${results['costs']:,.2f}")


if __name__ == "__main__":
    main()
