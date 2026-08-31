import pandas as pd


class InventoryVolatilityMarketMaker:
    """
    Basic inventory + volatility-adjusted market-making strategy.

    Logic:
    - Starts with a base spread around the mid-price.
    - Widens spread when rolling volatility increases.
    - Skews both bid and ask based on current inventory.
    - Stops quoting one side if max inventory limit is reached.
    """

    def __init__(
        self,
        base_spread_bps: float = 2.0,
        volatility_multiplier: float = 5000.0,
        inventory_skew: float = 2.0,
        order_size: float = 0.001,
        max_inventory: float = 0.01,
        fee_rate: float = 0.0002,
        starting_cash: float = 100_000.0,
    ):
        self.base_spread_bps = base_spread_bps
        self.volatility_multiplier = volatility_multiplier
        self.inventory_skew = inventory_skew
        self.order_size = order_size
        self.max_inventory = max_inventory
        self.fee_rate = fee_rate
        self.starting_cash = starting_cash

        self.cash = starting_cash
        self.inventory = 0.0
        self.fees_paid = 0.0
        self.bid_fills = 0
        self.ask_fills = 0

        self.history = []

    def calculate_quotes(self, row):
        mid = row["mid_price"]
        vol = row["rolling_volatility"]
        market_spread = row["market_spread"]

    # Start from the actual observed market spread
        base_spread = market_spread

    # Smaller volatility adjustment
        vol_spread = mid * vol * self.volatility_multiplier

        quoted_spread = base_spread + vol_spread

    # Inventory skew:
    # If long, shift quotes lower to encourage selling.
    # If short, shift quotes higher to encourage buying.
        skew = self.inventory_skew * self.inventory

        bid_quote = mid - quoted_spread / 2 - skew
        ask_quote = mid + quoted_spread / 2 - skew

        if bid_quote >= ask_quote:
            bid_quote = row["best_bid"]
            ask_quote = row["best_ask"]

        return bid_quote, ask_quote, quoted_spread  

    def run_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        for _, row in data.iterrows():
            timestamp = row["timestamp"]
            trade_price = row["trade_price"]
            mid = row["mid_price"]

            bid_quote, ask_quote, quoted_spread = self.calculate_quotes(row)

            placed_bid = self.inventory < self.max_inventory
            placed_ask = self.inventory > -self.max_inventory

            bid_filled = False
            ask_filled = False

            # Simplified fill logic:
            # If market trade price is at or below our bid, assume bid filled.
            if placed_bid and trade_price <= bid_quote:
                trade_value = bid_quote * self.order_size
                fee = trade_value * self.fee_rate

                self.cash -= trade_value + fee
                self.inventory += self.order_size
                self.fees_paid += fee
                self.bid_fills += 1
                bid_filled = True

            # If market trade price is at or above our ask, assume ask filled.
            if placed_ask and trade_price >= ask_quote:
                trade_value = ask_quote * self.order_size
                fee = trade_value * self.fee_rate

                self.cash += trade_value - fee
                self.inventory -= self.order_size
                self.fees_paid += fee
                self.ask_fills += 1
                ask_filled = True

            portfolio_value = self.cash + self.inventory * mid
            net_pnl = portfolio_value - self.starting_cash

            self.history.append(
                {
                    "timestamp": timestamp,
                    "mid_price": mid,
                    "trade_price": trade_price,
                    "rolling_volatility": row["rolling_volatility"],
                    "bid_quote": bid_quote,
                    "ask_quote": ask_quote,
                    "quoted_spread": quoted_spread,
                    "inventory": self.inventory,
                    "cash": self.cash,
                    "portfolio_value": portfolio_value,
                    "net_pnl": net_pnl,
                    "fees_paid": self.fees_paid,
                    "bid_filled": bid_filled,
                    "ask_filled": ask_filled,
                    "placed_bid": placed_bid,
                    "placed_ask": placed_ask,
                }
            )

        return pd.DataFrame(self.history)

    def summary(self, results: pd.DataFrame) -> dict:
        final_cash = self.cash
        final_mid_price = results["mid_price"].iloc[-1]
        final_inventory_value = self.inventory * final_mid_price
        final_portfolio_value = results["portfolio_value"].iloc[-1]
        final_net_pnl = final_portfolio_value - self.starting_cash

        running_max = results["portfolio_value"].cummax()
        drawdown = results["portfolio_value"] - running_max
        max_drawdown = drawdown.min()

        placed_bid_orders = results["placed_bid"].sum()
        placed_ask_orders = results["placed_ask"].sum()
        total_placed_orders = placed_bid_orders + placed_ask_orders
        total_fills = self.bid_fills + self.ask_fills
        fill_rate = (
            total_fills / total_placed_orders if total_placed_orders > 0 else 0
        )

        return {
            "final_cash": final_cash,
            "final_mid_price": final_mid_price,
            "final_inventory_value": final_inventory_value,
            "final_portfolio_value": final_portfolio_value,
            "final_net_pnl": final_net_pnl,
            "fees_paid": self.fees_paid,
            "bid_fills": self.bid_fills,
            "ask_fills": self.ask_fills,
            "total_fills": total_fills,
            "placed_bid_orders": placed_bid_orders,
            "placed_ask_orders": placed_ask_orders,
            "total_placed_orders": total_placed_orders,
            "fill_rate": fill_rate,
            "ending_inventory": self.inventory,
            "average_abs_inventory": results["inventory"].abs().mean(),
            "max_abs_inventory": results["inventory"].abs().max(),
            "max_drawdown": max_drawdown,
        }
