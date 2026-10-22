import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Initialize the Trading Client in Paper Mode (paper=True routes to sandbox)
# You can set these as environment variables or pass them directly
API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_PAPER_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "YOUR_PAPER_SECRET_KEY")

client = TradingClient(API_KEY, API_SECRET, paper=True)

# 1. Check your virtual paper account balance
account = client.get_account()
print(f"Paper Account Status: {account.status}")
print(f"Virtual Cash Available: ${account.cash}")

# 2. Example: Submit a test market order for 1 share of AAPL
order_data = MarketOrderRequest(
    symbol="AAPL",
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

# Uncomment below to actually place the paper trade order:
# market_order = client.submit_order(order_data=order_data)
# print(f"Paper order submitted successfully: {market_order.id}")