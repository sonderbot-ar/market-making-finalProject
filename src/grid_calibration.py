"""Memory-efficient summary evaluators for large parameter grids."""

import numpy as np


def calibration_arrays(data):
    return {
        column: data[column].to_numpy()
        for column in (
            "timestamp",
            "mid_price",
            "market_spread",
            "rolling_volatility",
            "rolling_return",
            "trade_price",
            "best_bid",
            "best_ask",
        )
    }


def _standard_summary(state, final_mid_price, sample_size):
    total_fills = state["bid_fills"] + state["ask_fills"]
    total_placed_orders = state["placed_bid_orders"] + state["placed_ask_orders"]
    final_inventory_value = state["inventory"] * final_mid_price
    final_portfolio_value = state["cash"] + final_inventory_value
    return {
        "final_cash": state["cash"],
        "final_mid_price": final_mid_price,
        "final_inventory_value": final_inventory_value,
        "final_portfolio_value": final_portfolio_value,
        "final_net_pnl": final_portfolio_value - 100_000.0,
        "fees_paid": state["fees_paid"],
        "bid_fills": state["bid_fills"],
        "ask_fills": state["ask_fills"],
        "total_fills": total_fills,
        "placed_bid_orders": state["placed_bid_orders"],
        "placed_ask_orders": state["placed_ask_orders"],
        "total_placed_orders": total_placed_orders,
        "fill_rate": total_fills / total_placed_orders if total_placed_orders else 0,
        "ending_inventory": state["inventory"],
        "average_abs_inventory": state["abs_inventory_sum"] / sample_size,
        "max_abs_inventory": state["max_abs_inventory"],
        "max_drawdown": state["max_drawdown"],
    }


def _initial_state():
    return {
        "cash": 100_000.0,
        "inventory": 0.0,
        "fees_paid": 0.0,
        "bid_fills": 0,
        "ask_fills": 0,
        "placed_bid_orders": 0,
        "placed_ask_orders": 0,
        "abs_inventory_sum": 0.0,
        "max_abs_inventory": 0.0,
        "running_max": -np.inf,
        "max_drawdown": 0.0,
    }


def _update_portfolio_state(state, mid):
    portfolio_value = state["cash"] + state["inventory"] * mid
    state["running_max"] = max(state["running_max"], portfolio_value)
    state["max_drawdown"] = min(
        state["max_drawdown"], portfolio_value - state["running_max"]
    )
    absolute_inventory = abs(state["inventory"])
    state["abs_inventory_sum"] += absolute_inventory
    state["max_abs_inventory"] = max(state["max_abs_inventory"], absolute_inventory)


def run_base_summary(arrays, volatility_multiplier, inventory_skew, max_inventory):
    state = _initial_state()
    order_size = 0.001
    fee_rate = 0.0002

    for mid, market_spread, volatility, trade_price, best_bid, best_ask in zip(
        arrays["mid_price"],
        arrays["market_spread"],
        arrays["rolling_volatility"],
        arrays["trade_price"],
        arrays["best_bid"],
        arrays["best_ask"],
    ):
        quoted_spread = market_spread + mid * volatility * volatility_multiplier
        skew = inventory_skew * state["inventory"]
        bid_quote = mid - quoted_spread / 2 - skew
        ask_quote = mid + quoted_spread / 2 - skew
        if bid_quote >= ask_quote:
            bid_quote, ask_quote = best_bid, best_ask

        placed_bid = state["inventory"] < max_inventory
        placed_ask = state["inventory"] > -max_inventory
        state["placed_bid_orders"] += placed_bid
        state["placed_ask_orders"] += placed_ask

        if placed_bid and trade_price <= bid_quote:
            trade_value = bid_quote * order_size
            fee = trade_value * fee_rate
            state["cash"] -= trade_value + fee
            state["inventory"] += order_size
            state["fees_paid"] += fee
            state["bid_fills"] += 1
        if placed_ask and trade_price >= ask_quote:
            trade_value = ask_quote * order_size
            fee = trade_value * fee_rate
            state["cash"] += trade_value - fee
            state["inventory"] -= order_size
            state["fees_paid"] += fee
            state["ask_fills"] += 1

        _update_portfolio_state(state, mid)

    return _standard_summary(state, arrays["mid_price"][-1], len(arrays["mid_price"]))


def run_improved_summary(
    arrays,
    volatility_multiplier,
    inventory_skew,
    trend_multiplier,
    trend_skew_cap_ratio,
    min_spread_safety_multiplier,
    max_allowed_volatility,
    liquidation_aggressiveness,
    liquidation_threshold=0.8,
    max_inventory=0.005,
):
    state = _initial_state()
    order_size = 0.001
    fee_rate = 0.0002
    paused_events = 0
    quoted_spread_sum = 0.0
    min_profitable_spread_sum = 0.0
    long_liquidation_activations = 0
    short_liquidation_activations = 0

    for (
        mid,
        market_spread,
        volatility,
        rolling_return,
        trade_price,
        best_bid,
        best_ask,
    ) in zip(
        arrays["mid_price"],
        arrays["market_spread"],
        arrays["rolling_volatility"],
        arrays["rolling_return"],
        arrays["trade_price"],
        arrays["best_bid"],
        arrays["best_ask"],
    ):
        quoted_spread = market_spread + mid * volatility * volatility_multiplier
        min_profitable_spread = mid * fee_rate * 2 * min_spread_safety_multiplier
        quoted_spread = max(quoted_spread, min_profitable_spread)
        skew = inventory_skew * state["inventory"]
        bid_quote = mid - quoted_spread / 2 - skew
        ask_quote = mid + quoted_spread / 2 - skew

        trend_skew = trend_multiplier * rolling_return * mid
        max_trend_skew = quoted_spread * trend_skew_cap_ratio
        trend_skew = max(-max_trend_skew, min(trend_skew, max_trend_skew))
        bid_quote += trend_skew
        ask_quote += trend_skew

        long_liquidation = state["inventory"] > liquidation_threshold * max_inventory
        short_liquidation = state["inventory"] < -liquidation_threshold * max_inventory
        long_liquidation_activations += long_liquidation
        short_liquidation_activations += short_liquidation
        if long_liquidation:
            ask_quote += liquidation_aggressiveness * (best_ask - ask_quote)
        if short_liquidation:
            bid_quote += liquidation_aggressiveness * (best_bid - bid_quote)
        if bid_quote >= ask_quote:
            bid_quote, ask_quote = best_bid, best_ask

        quoting_paused = (
            max_allowed_volatility is not None and volatility > max_allowed_volatility
        )
        paused_events += quoting_paused
        placed_bid = not quoting_paused and state["inventory"] < max_inventory
        placed_ask = not quoting_paused and state["inventory"] > -max_inventory
        state["placed_bid_orders"] += placed_bid
        state["placed_ask_orders"] += placed_ask

        if placed_bid and trade_price <= bid_quote:
            trade_value = bid_quote * order_size
            fee = trade_value * fee_rate
            state["cash"] -= trade_value + fee
            state["inventory"] += order_size
            state["fees_paid"] += fee
            state["bid_fills"] += 1
        if placed_ask and trade_price >= ask_quote:
            trade_value = ask_quote * order_size
            fee = trade_value * fee_rate
            state["cash"] += trade_value - fee
            state["inventory"] -= order_size
            state["fees_paid"] += fee
            state["ask_fills"] += 1

        quoted_spread_sum += quoted_spread
        min_profitable_spread_sum += min_profitable_spread
        _update_portfolio_state(state, mid)

    sample_size = len(arrays["mid_price"])
    summary = _standard_summary(state, arrays["mid_price"][-1], sample_size)
    summary.update(
        {
            "paused_events": paused_events,
            "pause_rate": paused_events / sample_size,
            "average_quoted_spread": quoted_spread_sum / sample_size,
            "average_min_profitable_spread": min_profitable_spread_sum / sample_size,
            "long_liquidation_activations": long_liquidation_activations,
            "short_liquidation_activations": short_liquidation_activations,
        }
    )
    return summary
