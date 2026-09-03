"""Inventory and volatility-adjusted market-making strategy.

The implementation remains in the original strategies package for backward
compatibility. This module provides the main-project-style import location.
"""

from src.strategies.inventory_volatility import InventoryVolatilityMarketMaker

__all__ = ["InventoryVolatilityMarketMaker"]
