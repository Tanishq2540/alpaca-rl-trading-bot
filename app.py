import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from alpaca.trading.client import TradingClient
from stable_baselines3 import PPO

from rl_trader import build_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ppo_trader.zip"
LOG_PATH = BASE_DIR / "trading.log"
HISTORY_PATH = BASE_DIR / "trade_history.csv"
SYMBOL = "AAPL"
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
DECISION_RE = re.compile(r"\b(BUY|SELL|HOLD)\b", re.IGNORECASE)

st.set_page_config(
    page_title="RL Trading Agent Monitor",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {font-weight: 600;}
    div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
    .signal-box {
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background: rgba(128, 128, 128, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_text(value):
    return getattr(value, "value", value)


@st.cache_data(ttl=20, show_spinner=False)
def fetch_alpaca(api_key, secret_key, paper):
    client = TradingClient(api_key, secret_key, paper=paper)
    account = client.get_account()
    account_data = {
        "status": str(as_text(account.status)),
        "equity": as_float(account.equity),
        "portfolio_value": as_float(account.portfolio_value),
        "cash": as_float(account.cash),
        "buying_power": as_float(account.buying_power),
        "currency": str(getattr(account, "currency", "USD")),
    }
    position_data = []
    for position in client.get_all_positions():
        position_data.append(
            {
                "Symbol": position.symbol,
                "Shares": as_float(position.qty),
                "Avg Entry": as_float(position.avg_entry_price),
                "Current Price": as_float(position.current_price),
                "Market Value": as_float(position.market_value),
                "Unrealized P/L": as_float(position.unrealized_pl),
                "Unrealized P/L %": as_float(position.unrealized_plpc) * 100,
            }
        )
    return account_data, position_data


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_features(symbol, lookback_days=400):
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        timeout=15,
    )
    if df is None or df.empty:
        raise ValueError(f"No market data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df.index.date < now.date()]
    if df.empty:
        raise ValueError(f"No completed daily bars available for {symbol}")
    features, prices, dates = build_features(df)
    return features, prices, dates


@st.cache_resource(show_spinner=False)
def load_model(path, modified_time):
    return PPO.load(path, device="cpu")


def get_signal():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.name}")
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    features, prices, dates = load_latest_features(SYMBOL)

    observation_size = int(np.prod(model.observation_space.shape))
    n_features = int(features.shape[1])
    if observation_size % n_features != 0:
        raise ValueError("Model observation size does not match the feature set")
    window = observation_size // n_features
    if len(features) < window:
        raise ValueError(f"Need at least {window} feature rows, got {len(features)}")

    observation = features[-window:].reshape(-1).astype(np.float32)
    action, _ = model.predict(observation, deterministic=True)
    action = int(action)

    probabilities = None
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        distribution = model.policy.get_distribution(obs_tensor)
        probabilities = (
            distribution.distribution.probs.detach().cpu().numpy().ravel().tolist()
        )
    except Exception:
        probabilities = None

    return {
        "action": action,
        "action_name": ACTION_NAMES.get(action, str(action)),
        "probabilities": probabilities,
        "last_bar": dates[-1].date(),
        "last_close": float(prices[-1]),
        "window": window,
        "features": n_features,
    }


def parse_log(path, modified_time):
    if not path.exists():
        return None, []
    with path.open("r", errors="replace") as handle:
        lines = handle.readlines()[-500:]
    rows = []
    for line in lines:
        decision = DECISION_RE.search(line)
        if not decision:
            continue
        timestamp = TIMESTAMP_RE.search(line)
        rows.append(
            {
                "Timestamp": timestamp.group(0) if timestamp else "—",
                "Decision": decision.group(1).upper(),
                "Details": line.strip()[:300],
            }
        )
    return lines, rows[-25:][::-1]


@st.cache_data(ttl=30, show_spinner=False)
def load_trade_history(path, modified_time):
    history = pd.read_csv(path)
    missing = {"Timestamp", "Action", "Portfolio_Value"} - set(history.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    history["Timestamp"] = pd.to_datetime(
        history["Timestamp"], utc=True, errors="coerce"
    )
    history["Portfolio_Value"] = pd.to_numeric(
        history["Portfolio_Value"], errors="coerce"
    )
    history = history.dropna(subset=["Timestamp", "Portfolio_Value"])
    return history.sort_values("Timestamp").reset_index(drop=True)


st.title("RL Trading Agent Monitor")
st.caption(f"{SYMBOL} · PPO policy · Alpaca paper account read-only dashboard")

with st.sidebar:
    st.header("Connection")
    env_key = os.environ.get("ALPACA_API_KEY", "")
    env_secret = os.environ.get("ALPACA_SECRET_KEY", "")

    with st.form("alpaca_credentials"):
        key_input = st.text_input(
            "ALPACA_API_KEY",
            value=env_key,
            type="password",
            help="Falls back to the ALPACA_API_KEY environment variable.",
        )
        secret_input = st.text_input(
            "ALPACA_SECRET_KEY",
            value=env_secret,
            type="password",
            help="Falls back to the ALPACA_SECRET_KEY environment variable.",
        )
        paper_mode = st.toggle(
            "Live Paper Trading Mode",
            value=True,
            help="Off switches the read-only dashboard to the live account endpoint.",
        )
        apply_clicked = st.form_submit_button(
            "Apply credentials", use_container_width=True
        )

    if apply_clicked:
        st.session_state["alpaca_creds"] = {
            "key": key_input,
            "secret": secret_input,
            "paper": paper_mode,
        }
        st.rerun()

    if "alpaca_creds" not in st.session_state and env_key and env_secret:
        st.session_state["alpaca_creds"] = {
            "key": env_key,
            "secret": env_secret,
            "paper": True,
        }

    credentials = st.session_state.get("alpaca_creds")
    if credentials:
        mode_label = "Paper" if credentials["paper"] else "Live (read-only)"
        st.caption(f"Mode: **{mode_label}**")

    st.divider()
    st.caption(
        f"Model: {'available' if MODEL_PATH.exists() else 'missing'} · "
        f"Log: {'available' if LOG_PATH.exists() else 'missing'} · "
        f"History: {'available' if HISTORY_PATH.exists() else 'missing'}"
    )
    if st.button("Refresh data", use_container_width=True):
        fetch_alpaca.clear()
        load_latest_features.clear()
        st.rerun()
    st.caption("Keys are kept in memory for this session and are never written to disk.")

st.subheader("Account Overview")

if not credentials or not credentials["key"] or not credentials["secret"]:
    st.info(
        "Add your Alpaca paper API key and secret in the sidebar, or set "
        "`ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in the environment."
    )
else:
    try:
        with st.spinner("Connecting to Alpaca…"):
            account, positions = fetch_alpaca(
                credentials["key"], credentials["secret"], credentials["paper"]
            )
    except Exception as exc:
        st.error(f"Unable to reach Alpaca: {exc}")
    else:
        if not credentials["paper"]:
            st.warning(
                "Live account endpoint selected. This dashboard is read-only and "
                "never submits orders."
            )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Account Balance", f"${account['equity']:,.2f}")
        metric_cols[1].metric(
            "Portfolio Value", f"${account['portfolio_value']:,.2f}"
        )
        metric_cols[2].metric("Cash", f"${account['cash']:,.2f}")
        metric_cols[3].metric("Status", str(account["status"]).title())

        st.markdown(f"**Open {SYMBOL} position**")
        aapl_positions = [p for p in positions if p["Symbol"] == SYMBOL]
        if aapl_positions:
            position = aapl_positions[0]
            pos_cols = st.columns(4)
            pos_cols[0].metric("Shares Held", f"{position['Shares']:,.4f}")
            pos_cols[1].metric(
                "Market Value", f"${position['Market Value']:,.2f}"
            )
            pos_cols[2].metric(
                "Unrealized P/L",
                f"${position['Unrealized P/L']:,.2f}",
                f"{position['Unrealized P/L %']:+.2f}%",
            )
            pos_cols[3].metric(
                "Current Price", f"${position['Current Price']:,.2f}"
            )
            table = pd.DataFrame(aapl_positions)
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Shares": st.column_config.NumberColumn(format="%.4f"),
                    "Avg Entry": st.column_config.NumberColumn(format="$%.2f"),
                    "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                    "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
                    "Unrealized P/L %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                },
            )
        else:
            st.info(f"No open {SYMBOL} position.")

        if len(positions) > len(aapl_positions):
            with st.expander("All open positions"):
                st.dataframe(
                    pd.DataFrame(positions),
                    use_container_width=True,
                    hide_index=True,
                )

st.divider()
st.subheader("Model Performance")
perf_cols = st.columns(3)
perf_cols[0].metric("OOS Return", "+37.23%")
perf_cols[1].metric("Max Drawdown", "-25.35%")
perf_cols[2].metric("Trades", "72")
st.caption(
    "Recorded 50k-step out-of-sample backtest for AAPL "
    "(2023-01-03 to 2024-12-31). Historical results are not live performance."
)

st.divider()
st.subheader("Agent Decision Log")
log_lines = []
log_rows = []
if LOG_PATH.exists():
    log_lines, log_rows = parse_log(LOG_PATH, LOG_PATH.stat().st_mtime)
if not LOG_PATH.exists():
    st.info(
        "`trading.log` was not found. The log will appear here once the agent "
        "writes decisions to it."
    )
elif log_rows:
    st.dataframe(
        pd.DataFrame(log_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.TextColumn(width="medium"),
            "Decision": st.column_config.TextColumn(width="small"),
            "Details": st.column_config.TextColumn(width="large"),
        },
    )
else:
    st.info("`trading.log` exists, but no BUY/SELL/HOLD decisions were found.")
    if log_lines:
        st.code("".join(log_lines[-20:]), language="text")

st.divider()
st.subheader("Live Trade History")
if not HISTORY_PATH.exists():
    st.warning(
        "`trade_history.csv` was not found. It will appear here after "
        "`alpaca_live_bot.py` records its first daily decision."
    )
else:
    try:
        history = load_trade_history(
            str(HISTORY_PATH), HISTORY_PATH.stat().st_mtime
        )
    except Exception as exc:
        st.error(f"Could not read `trade_history.csv`: {exc}")
    else:
        if history.empty:
            st.warning("`trade_history.csv` exists but has no valid rows yet.")
        else:
            latest = history.iloc[-1]
            history_cols = st.columns(3)
            history_cols[0].metric(
                "Latest Portfolio Value",
                f"${latest['Portfolio_Value']:,.2f}",
            )
            history_cols[1].metric("Latest Action", str(latest["Action"]).title())
            history_cols[2].metric("Records", f"{len(history):,}")

            chart_data = history.set_index("Timestamp")["Portfolio_Value"]
            st.line_chart(chart_data, height=320)
            st.caption("Portfolio value recorded by the daily paper-trading bot.")

            recent_history = history.tail(20).iloc[::-1].copy()
            recent_history["Timestamp"] = recent_history["Timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            st.dataframe(
                recent_history[["Timestamp", "Action", "Portfolio_Value"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Timestamp": st.column_config.TextColumn(width="medium"),
                    "Action": st.column_config.TextColumn(width="small"),
                    "Portfolio_Value": st.column_config.NumberColumn(
                        "Portfolio Value", format="$%.2f"
                    ),
                },
            )

st.divider()
st.subheader("Manual Model Check")
st.caption(
    "Runs `ppo_trader.zip` on the latest completed daily AAPL bars. "
    "Dry run only — no orders are submitted."
)
if st.button("Run dry-run evaluation", type="primary"):
    try:
        with st.spinner("Loading model and latest market data…"):
            signal = get_signal()
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
    else:
        st.success(f"Model action: {signal['action_name']}")
        st.markdown(
            f"<div class='signal-box'>"
            f"Last completed bar: <b>{signal['last_bar']}</b> · "
            f"Close: <b>${signal['last_close']:,.2f}</b> · "
            f"Window: <b>{signal['window']}</b> bars · "
            f"Features: <b>{signal['features']}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if signal["probabilities"]:
            prob_cols = st.columns(3)
            for column, name, probability in zip(
                prob_cols, ["HOLD", "BUY", "SELL"], signal["probabilities"]
            ):
                column.metric(f"{name} probability", f"{probability * 100:.1f}%")
        st.caption("Dry run complete. No order was submitted.")
