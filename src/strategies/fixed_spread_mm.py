import pandas as pd

class FixedSpreadMM:

    def __init__(self, half_spread: float, trade_size: float = 0.001, 
                 max_inventory: float = 0.005, fee_rate: float = 0.0002, 
                 starting_cash: float = 100000.0):
        
        self.half_spread = half_spread
        self.trade_size = trade_size
        self.max_inventory = max_inventory
        self.fee_rate = fee_rate
        self.inventory = 0.0
        self.cash = starting_cash
        self.initial_portfolio_value = starting_cash
        self.bid_fills = 0
        self.ask_fills = 0
        self.fees_paid = 0.0
        self.current_bid = None
        self.current_ask = None


    def update_quotes(self, mid_price: float):
        epsilon = 1e-9

        if self.inventory + self.trade_size <= self.max_inventory + epsilon:
            self.current_bid = mid_price - self.half_spread
        else:
            self.current_bid = None

        if self.inventory - self.trade_size >= -self.max_inventory - epsilon:
            self.current_ask = mid_price + self.half_spread
        else:
            self.current_ask = None
        

    def process_trade(self, market_price: float):
        if self.current_bid is not None and market_price <= self.current_bid:
            notional_value = self.current_bid * self.trade_size
            fee = notional_value * self.fee_rate
            
            self.inventory += self.trade_size
            self.cash -= (notional_value + fee)
            
            self.bid_fills += 1
            self.fees_paid += fee
        
        elif self.current_ask is not None and market_price >= self.current_ask:
            notional_value = self.current_ask * self.trade_size
            fee = notional_value * self.fee_rate
            
            self.inventory -= self.trade_size
            self.cash += (notional_value - fee)
            
            self.ask_fills += 1
            self.fees_paid += fee

    def get_portfolio_value(self, mid_price: float) -> float:
        return self.cash + (self.inventory * mid_price)

    def calculate_pnl(self, mid_price: float) -> float:
        return self.get_portfolio_value(mid_price) - self.initial_portfolio_value