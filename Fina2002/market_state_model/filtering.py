from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm_model import FittedMarketStateModel


def forward_filter_probabilities(model: FittedMarketStateModel, factors: pd.DataFrame) -> pd.DataFrame:
    x = factors.loc[:, model.factor_columns].to_numpy(dtype=float)
    b = model.hmm.emission_prob(x)
    a = model.hmm.transmat_
    start = model.hmm.startprob_
    n, k = b.shape
    alpha = np.zeros((n, k), dtype=float)
    xi = np.zeros((n, k, k), dtype=float)

    alpha[0] = start * b[0]
    alpha[0] /= max(alpha[0].sum(), 1e-300)
    for t in range(1, n):
        joint = alpha[t - 1, :, None] * a * b[t, None, :]
        denom = joint.sum()
        if denom <= 0:
            joint = np.full_like(joint, 1.0 / joint.size)
            denom = joint.sum()
        xi[t] = joint / denom
        alpha[t] = xi[t].sum(axis=0)
        alpha[t] /= max(alpha[t].sum(), 1e-300)

    h = model.label_to_raw["H"]
    bull = model.label_to_raw["L+"]
    bear = model.label_to_raw["L-"]
    label_order = {raw: label for raw, label in model.raw_to_label.items()}
    regimes = [label_order[int(raw)] for raw in np.argmax(alpha, axis=1)]

    out = pd.DataFrame(
        {
            "date": factors["date"].values,
            "p_high_entropy": alpha[:, h],
            "p_low_bull": alpha[:, bull],
            "p_low_bear": alpha[:, bear],
            "posterior_entropy": -(alpha * np.log(np.maximum(alpha, 1e-300))).sum(axis=1),
            "xi_H_to_Lplus": xi[:, h, bull],
            "xi_H_to_Lminus": xi[:, h, bear],
            "market_regime": regimes,
        }
    )
    return out


def smoothed_probabilities_for_diagnostics(model: FittedMarketStateModel, factors: pd.DataFrame) -> pd.DataFrame:
    x = factors.loc[:, model.factor_columns].to_numpy(dtype=float)
    gamma = model.hmm.smooth_probabilities(x)
    h = model.label_to_raw["H"]
    bull = model.label_to_raw["L+"]
    bear = model.label_to_raw["L-"]
    return pd.DataFrame(
        {
            "date": factors["date"].values,
            "smooth_p_high_entropy": gamma[:, h],
            "smooth_p_low_bull": gamma[:, bull],
            "smooth_p_low_bear": gamma[:, bear],
        }
    )

