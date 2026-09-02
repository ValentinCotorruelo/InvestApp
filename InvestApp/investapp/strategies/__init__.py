from .base import Evaluation, ExitInfo, Signal, Strategy
from .trend_follower import TrendFollowerStrategy
from .mean_reversion import MeanReversionStrategy
from .confluence import ConfluenceStrategy
from .hybrid import HybridStrategy
from .support_resistance_strategy import SupportResistanceStrategy

STRATEGY_REGISTRY = {
    "trend": TrendFollowerStrategy,
    "trend_follower": TrendFollowerStrategy,
    "mean_reversion": MeanReversionStrategy,
    "confluence": ConfluenceStrategy,
    "hybrid": HybridStrategy,
    "support_resistance": SupportResistanceStrategy,
    "sr": SupportResistanceStrategy,
}

STRATEGIES = [
    TrendFollowerStrategy.name,
    MeanReversionStrategy.name,
    ConfluenceStrategy.name,
    HybridStrategy.name,
    SupportResistanceStrategy.name,
]


def get_strategy(name: str, *args, **kwargs) -> Strategy:
    """Instancia una estrategia por nombre."""
    key = name.strip().lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Estrategia desconocida '{name}'. Disponibles: {sorted(set(STRATEGY_REGISTRY))}"
        )
    return STRATEGY_REGISTRY[key](*args, **kwargs)


__all__ = [
    "Evaluation",
    "ExitInfo",
    "Signal",
    "Strategy",
    "TrendFollowerStrategy",
    "MeanReversionStrategy",
    "ConfluenceStrategy",
    "HybridStrategy",
    "SupportResistanceStrategy",
    "get_strategy",
    "STRATEGY_REGISTRY",
    "STRATEGIES",
]