import pandas as pd

class FixedSpreadMM:

    def __init__(self, half_spread: float, trade_size: float = 0.01):
        self.half_spread = half_spread
        self.trade_size = trade_size
        self.inventory = 0.0
        self.cash = 0.0
        self.current_bid = None
        self.current_ask = None

    def update_quotes(self, mid_price: float):
        self.current_bid = mid_price - self.half_spread
        self.current_ask = mid_price + self.half_spread

    def process_trade(self, market_price: float):
        if self.current_bid is None or self.current_ask is None:
            return 
        
        if market_price <= self.current_bid:
            self.inventory += self.trade_size
            self.cash -= self.current_bid * self.trade_size

        elif market_price >= self.current_ask:
            self.inventory -= self.trade_size
            self.cash += self.current_ask * self.trade_size

    def calculate_pnl(self, mid_price: float) -> float:
        return self.cash + (self.inventory * mid_price)