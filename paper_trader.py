from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Initialize client with paper-trading keys
# (Pass your keys directly or set them as environment variables)
client = TradingClient('YOUR_PAPER_API_KEY', 'YOUR_PAPER_SECRET_KEY', paper=True)

# Check your virtual account balance
account = client.get_account()
print(f"Virtual Cash Available: ${account.cash}")

# Example: Submit a test market order to buy 1 share of AAPL in paper trading
order_data = MarketOrderRequest(
    symbol="AAPL",
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

# market_order = client.submit_order(order_data=order_data)
# print(f"Order submitted: {market_order.id}")