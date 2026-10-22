import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from stable_baselines3 import PPO

from rl_trader import build_features, fetch_data

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def get_credentials():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        )
    return api_key, secret_key


def load_market_data(symbol, window, lookback_days, drop_partial_bar):
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    df = fetch_data(symbol, start, end)
    if drop_partial_bar:
        df = df[df.index.date < now.date()]
    features, prices, dates = build_features(df)
    if len(features) < window:
        raise SystemExit(
            f"Not enough data for {symbol}: need {window} rows, got {len(features)}"
        )
    return features, prices, dates


def get_position(client, symbol):
    for position in client.get_all_positions():
        if position.symbol == symbol:
            return position
    return None


def main():
    parser = argparse.ArgumentParser(description="PPO paper-trading bot for Alpaca")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--model", default="ppo_trader.zip")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--lookback-days", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true", help="print the signal only")
    parser.add_argument(
        "--force", action="store_true", help="submit even when the market is closed"
    )
    parser.add_argument(
        "--include-today",
        action="store_true",
        help="include the current (possibly partial) daily bar",
    )
    args = parser.parse_args()

    api_key, secret_key = get_credentials()
    client = TradingClient(api_key, secret_key, paper=True)

    try:
        account = client.get_account()
        clock = client.get_clock()
    except Exception as exc:
        raise SystemExit(f"Alpaca connection failed: {exc}")

    model = PPO.load(args.model, device="cpu")
    if int(model.action_space.n) != 3:
        raise SystemExit("Expected a model with 3 actions (hold, buy, sell)")

    features, prices, dates = load_market_data(
        args.symbol,
        args.window,
        args.lookback_days,
        drop_partial_bar=clock.is_open and not args.include_today,
    )
    observation = features[-args.window :].reshape(-1).astype(np.float32)
    expected_shape = tuple(model.observation_space.shape)
    if observation.shape != expected_shape:
        raise SystemExit(
            f"Observation shape mismatch: model expects {expected_shape}, "
            f"got {observation.shape}. Retrain or align the feature set."
        )

    action, _ = model.predict(observation, deterministic=True)
    action = int(action)
    position = get_position(client, args.symbol)
    held_qty = float(position.qty) if position else 0.0
    last_close = float(prices[-1])

    print(f"Alpaca paper account: status={account.status}, equity=${float(account.equity):,.2f}, cash=${float(account.cash):,.2f}")
    print(f"Market open: {clock.is_open} (next open: {clock.next_open})")
    print(f"{args.symbol} last bar: {dates[-1].date()} close=${last_close:,.2f}")
    print(f"Position: {held_qty:g} shares")
    print(f"Model action: {ACTION_NAMES[action]} ({action})")

    if action == 1 and held_qty == 0.0:
        notional = round(min(float(account.cash), float(account.buying_power)) * 0.99, 2)
        if notional < 1.0:
            print("Not enough buying power to open a position; nothing to do.")
            return
        print(f"Signal: BUY ${notional:,.2f} of {args.symbol}")
        if args.dry_run:
            print("Dry run: order not submitted.")
            return
        if not clock.is_open and not args.force:
            print("Market is closed; skipping order. Use --force to queue it.")
            return
        order = client.submit_order(
            MarketOrderRequest(
                symbol=args.symbol,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        print(f"Submitted BUY order: id={order.id} status={order.status}")
    elif action == 2 and held_qty > 0.0:
        print(f"Signal: SELL {held_qty:g} shares of {args.symbol}")
        if args.dry_run:
            print("Dry run: order not submitted.")
            return
        if not clock.is_open and not args.force:
            print("Market is closed; skipping order. Use --force to queue it.")
            return
        order = client.close_position(args.symbol)
        print(f"Submitted SELL/close order: id={order.id} status={order.status}")
    else:
        print("Signal: HOLD; no order needed.")


if __name__ == "__main__":
    main()
