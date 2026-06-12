from __future__ import annotations

import pandas as pd

from .config import MarketStateConfig


def add_transition_features(probabilities: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    out = probabilities.copy().sort_values("date")
    out["delta_p_low_bull"] = out["p_low_bull"].diff().fillna(0.0)
    out["delta_p_high_entropy"] = out["p_high_entropy"].diff().fillna(0.0)
    out["delta_p_low_bear"] = out["p_low_bear"].diff().fillna(0.0)
    out["transition_score"] = out["xi_H_to_Lplus"] - out["xi_H_to_Lminus"]
    out["bullish_transition_signal"] = (
        (out["p_low_bull"] > config.bull_prob_min)
        & (out["delta_p_low_bull"] > config.bull_delta_min)
        & (out["delta_p_high_entropy"] < config.entropy_delta_max)
        & (out["delta_p_low_bear"] <= config.bear_delta_max)
    )
    out["bearish_transition_signal"] = (
        (out["p_low_bear"] > config.bear_prob_min)
        & (out["delta_p_low_bear"] > config.bear_delta_min)
        & (out["delta_p_high_entropy"] < config.entropy_delta_max)
        & (out["delta_p_low_bull"] <= config.bull_delta_max)
    )
    return out

