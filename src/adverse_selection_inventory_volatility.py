import pandas as pd

from src.inventory_volatility import InventoryVolatilityMarketMaker


class AdverseSelectionAwareInventoryVolatilityMarketMaker(
    InventoryVolatilityMarketMaker
):
    """Inventory-volatility market maker with adverse-selection controls."""

    def __init__(
        self,
        base_spread_bps: float = 2.0,
        volatility_multiplier: float = 5000.0,
        inventory_skew: float = 2.0,
        trend_multiplier: float = 1.0,
        trend_skew_cap_ratio: float = 0.5,
        min_spread_safety_multiplier: float = 1.5,
        volatility_pause_quantile: float | None = None,
        max_allowed_volatility: float | None = None,
        liquidation_threshold: float = 0.8,
        liquidation_aggressiveness: float = 0.5,
        order_size: float = 0.001,
        max_inventory: float = 0.01,
        fee_rate: float = 0.0002,
        starting_cash: float = 100_000.0,
    ):
        super().__init__(
            base_spread_bps=base_spread_bps,
            volatility_multiplier=volatility_multiplier,
            inventory_skew=inventory_skew,
            order_size=order_size,
            max_inventory=max_inventory,
            fee_rate=fee_rate,
            starting_cash=starting_cash,
        )
        self.trend_multiplier = trend_multiplier
        self.trend_skew_cap_ratio = trend_skew_cap_ratio
        self.min_spread_safety_multiplier = min_spread_safety_multiplier
        self.volatility_pause_quantile = volatility_pause_quantile
        self.max_allowed_volatility = max_allowed_volatility
        self.liquidation_threshold = liquidation_threshold
        self.liquidation_aggressiveness = liquidation_aggressiveness

    def calculate_quotes(self, row):
        mid = row["mid_price"]
        volatility = row["rolling_volatility"]
        market_spread = row["market_spread"]

        volatility_spread = mid * volatility * self.volatility_multiplier
        quoted_spread = market_spread + volatility_spread

        min_profitable_spread = (
            mid * self.fee_rate * 2 * self.min_spread_safety_multiplier
        )
        quoted_spread = max(quoted_spread, min_profitable_spread)

        inventory_skew = self.inventory_skew * self.inventory
        bid_quote = mid - quoted_spread / 2 - inventory_skew
        ask_quote = mid + quoted_spread / 2 - inventory_skew

        rolling_return = row.get("rolling_return", 0.0)
        trend_skew = self.trend_multiplier * rolling_return * mid
        max_trend_skew = quoted_spread * self.trend_skew_cap_ratio
        trend_skew = max(-max_trend_skew, min(trend_skew, max_trend_skew))
        bid_quote += trend_skew
        ask_quote += trend_skew

        if self.inventory > self.liquidation_threshold * self.max_inventory:
            ask_quote += self.liquidation_aggressiveness * (row["best_ask"] - ask_quote)
        if self.inventory < -self.liquidation_threshold * self.max_inventory:
            bid_quote += self.liquidation_aggressiveness * (row["best_bid"] - bid_quote)

        if bid_quote >= ask_quote:
            bid_quote = row["best_bid"]
            ask_quote = row["best_ask"]

        return (
            bid_quote,
            ask_quote,
            quoted_spread,
            trend_skew,
            min_profitable_spread,
        )

    def run_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        for _, row in data.iterrows():
            timestamp = row["timestamp"]
            trade_price = row["trade_price"]
            mid = row["mid_price"]
            volatility = row["rolling_volatility"]

            (
                bid_quote,
                ask_quote,
                quoted_spread,
                trend_skew,
                min_profitable_spread,
            ) = self.calculate_quotes(row)

            quoting_paused = (
                self.max_allowed_volatility is not None
                and volatility > self.max_allowed_volatility
            )
            long_liquidation_active = (
                self.inventory > self.liquidation_threshold * self.max_inventory
            )
            short_liquidation_active = (
                self.inventory < -self.liquidation_threshold * self.max_inventory
            )
            placed_bid = not quoting_paused and self.inventory < self.max_inventory
            placed_ask = not quoting_paused and self.inventory > -self.max_inventory

            bid_filled = False
            ask_filled = False

            if placed_bid and trade_price <= bid_quote:
                trade_value = bid_quote * self.order_size
                fee = trade_value * self.fee_rate
                self.cash -= trade_value + fee
                self.inventory += self.order_size
                self.fees_paid += fee
                self.bid_fills += 1
                bid_filled = True

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
            inventory_ratio = (
                self.inventory / self.max_inventory if self.max_inventory else 0.0
            )

            self.history.append(
                {
                    "timestamp": timestamp,
                    "mid_price": mid,
                    "trade_price": trade_price,
                    "rolling_volatility": volatility,
                    "rolling_return": row.get("rolling_return", 0.0),
                    "bid_quote": bid_quote,
                    "ask_quote": ask_quote,
                    "quoted_spread": quoted_spread,
                    "trend_skew": trend_skew,
                    "min_profitable_spread": min_profitable_spread,
                    "inventory": self.inventory,
                    "inventory_ratio": inventory_ratio,
                    "cash": self.cash,
                    "portfolio_value": portfolio_value,
                    "net_pnl": net_pnl,
                    "fees_paid": self.fees_paid,
                    "bid_filled": bid_filled,
                    "ask_filled": ask_filled,
                    "placed_bid": placed_bid,
                    "placed_ask": placed_ask,
                    "quoting_paused": quoting_paused,
                    "long_liquidation_active": long_liquidation_active,
                    "short_liquidation_active": short_liquidation_active,
                }
            )

        return pd.DataFrame(self.history)

    def summary(self, results: pd.DataFrame) -> dict:
        summary = super().summary(results)
        paused_events = results["quoting_paused"].sum()
        summary.update(
            {
                "paused_events": paused_events,
                "pause_rate": paused_events / len(results),
                "average_quoted_spread": results["quoted_spread"].mean(),
                "average_min_profitable_spread": results[
                    "min_profitable_spread"
                ].mean(),
                "long_liquidation_activations": results[
                    "long_liquidation_active"
                ].sum(),
                "short_liquidation_activations": results[
                    "short_liquidation_active"
                ].sum(),
            }
        )
        return summary
