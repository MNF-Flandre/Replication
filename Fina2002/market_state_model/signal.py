from __future__ import annotations

import pandas as pd

from .config import MarketStateConfig


def add_entry_score(transitions: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    out = transitions.copy()
    out["entry_score"] = (
        out["p_low_bull"]
        + config.entry_gamma * out["delta_p_low_bull"]
        - config.entry_rho * out["p_low_bear"]
        - config.entry_xi * out["p_high_entropy"]
        - config.entry_omega * out["posterior_entropy"]
    )
    return out

