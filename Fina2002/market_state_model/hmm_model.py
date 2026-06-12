from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MarketStateConfig

try:
    from qlib_framework.rust_accel import fit_diagonal_gaussian_hmm_seeded as rust_fit_hmm_seeded
    from qlib_framework.rust_accel import rust_available
except ImportError:
    rust_fit_hmm_seeded = None

    def rust_available() -> bool:
        return False


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    row_sum = values.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return values / row_sum


class DiagonalGaussianHMM:
    def __init__(self, n_states: int, n_features: int, config: MarketStateConfig):
        self.n_states = n_states
        self.n_features = n_features
        self.config = config
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.monitor_: list[float] = []

    def _initial_labels(self, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        e_idx = columns.index("E")
        d_idx = columns.index("D")
        b_idx = columns.index("B")
        e = x[:, e_idx]
        directional = x[:, d_idx] + 0.5 * x[:, b_idx]
        labels = np.zeros(len(x), dtype=int)
        high_cut = np.nanquantile(e, 2 / 3)
        labels[e >= high_cut] = 0
        remaining = labels != 0
        if remaining.any():
            direction_cut = np.nanmedian(directional[remaining])
            labels[remaining & (directional >= direction_cut)] = 1
            labels[remaining & (directional < direction_cut)] = 2
        for state in range(self.n_states):
            if not np.any(labels == state):
                labels[np.argsort(e)[state :: self.n_states]] = state
        return labels

    def _initialize_from_labels(self, x: np.ndarray, labels: np.ndarray) -> None:
        start = np.bincount([labels[0]], minlength=self.n_states).astype(float) + self.config.startprob_prior
        self.startprob_ = start / start.sum()

        trans = np.full((self.n_states, self.n_states), self.config.transmat_prior, dtype=float)
        for prev, cur in zip(labels[:-1], labels[1:]):
            trans[prev, cur] += 1.0
        self.transmat_ = _normalize_rows(trans)

        means = np.zeros((self.n_states, self.n_features), dtype=float)
        covars = np.zeros_like(means)
        global_mean = np.nanmean(x, axis=0)
        global_var = np.nanvar(x, axis=0) + self.config.covariance_floor
        for state in range(self.n_states):
            mask = labels == state
            if mask.any():
                means[state] = x[mask].mean(axis=0)
                covars[state] = x[mask].var(axis=0) + self.config.covariance_floor
            else:
                means[state] = global_mean
                covars[state] = global_var
        self.means_ = means
        self.covars_ = np.maximum(covars, self.config.covariance_floor)

    def _initialize(self, x: np.ndarray, columns: tuple[str, ...]) -> None:
        labels = self._initial_labels(x, columns)
        self._initialize_from_labels(x, labels)

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.covars_ is None:
            raise RuntimeError("Model parameters are not initialized.")
        diff = x[:, None, :] - self.means_[None, :, :]
        log_det = np.log(self.covars_).sum(axis=1)
        quad = ((diff**2) / self.covars_[None, :, :]).sum(axis=2)
        return -0.5 * (self.n_features * np.log(2.0 * np.pi) + log_det[None, :] + quad)

    def emission_prob(self, x: np.ndarray) -> np.ndarray:
        log_prob = self._log_emission(x)
        shifted = log_prob - np.max(log_prob, axis=1, keepdims=True)
        return np.exp(shifted) + 1e-300

    def _forward_backward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.startprob_ is None or self.transmat_ is None:
            raise RuntimeError("Model parameters are not initialized.")
        b = self.emission_prob(x)
        n = len(x)
        alpha = np.zeros((n, self.n_states), dtype=float)
        beta = np.zeros_like(alpha)
        scale = np.zeros(n, dtype=float)

        alpha[0] = self.startprob_ * b[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= max(scale[0], 1e-300)
        for t in range(1, n):
            alpha[t] = alpha[t - 1] @ self.transmat_ * b[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= max(scale[t], 1e-300)

        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = self.transmat_ @ (b[t + 1] * beta[t + 1])
            beta[t] /= max(scale[t + 1], 1e-300)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        loglik = float(np.log(np.maximum(scale, 1e-300)).sum())
        return gamma, alpha, beta, loglik

    def fit(self, x: np.ndarray, columns: tuple[str, ...]) -> "DiagonalGaussianHMM":
        if len(x) < self.n_states * 20:
            raise ValueError("Not enough observations to fit a three-state HMM.")
        x = np.ascontiguousarray(x, dtype=float)
        labels = self._initial_labels(x, columns)
        self.monitor_ = []

        if rust_fit_hmm_seeded is not None and rust_available():
            try:
                start, trans, means, covars, monitor = rust_fit_hmm_seeded(
                    x,
                    labels,
                    n_states=self.n_states,
                    max_iter=self.config.hmm_max_iter,
                    tol=self.config.hmm_tol,
                    start_prior=self.config.startprob_prior,
                    trans_prior=self.config.transmat_prior,
                    covariance_floor=self.config.covariance_floor,
                )
                self.startprob_ = start
                self.transmat_ = trans
                self.means_ = means
                self.covars_ = covars
                self.monitor_ = monitor
                return self
            except Exception:
                self.monitor_ = []

        self._initialize_from_labels(x, labels)
        last_loglik = -np.inf

        for _ in range(self.config.hmm_max_iter):
            gamma, alpha, beta, loglik = self._forward_backward(x)
            b = self.emission_prob(x)

            xi_sum = np.full((self.n_states, self.n_states), self.config.transmat_prior, dtype=float)
            for t in range(len(x) - 1):
                numer = alpha[t, :, None] * self.transmat_ * b[t + 1, None, :] * beta[t + 1, None, :]
                denom = numer.sum()
                if denom > 0:
                    xi_sum += numer / denom

            weights = gamma.sum(axis=0) + 1e-12
            means = (gamma.T @ x) / weights[:, None]
            covars = np.zeros_like(means)
            for state in range(self.n_states):
                diff = x - means[state]
                covars[state] = (gamma[:, state][:, None] * diff**2).sum(axis=0) / weights[state]

            start = gamma[0] + self.config.startprob_prior
            self.startprob_ = start / start.sum()
            self.transmat_ = _normalize_rows(xi_sum)
            self.means_ = means
            self.covars_ = np.maximum(covars, self.config.covariance_floor)
            self.monitor_.append(loglik)

            if np.isfinite(last_loglik) and abs(loglik - last_loglik) < self.config.hmm_tol:
                break
            last_loglik = loglik
        return self

    def smooth_probabilities(self, x: np.ndarray) -> np.ndarray:
        gamma, _, _, _ = self._forward_backward(x)
        return gamma


@dataclass
class FittedMarketStateModel:
    hmm: DiagonalGaussianHMM
    factor_columns: tuple[str, ...]
    raw_to_label: dict[int, str]
    label_to_raw: dict[str, int]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    state_table: pd.DataFrame

    def to_parameter_dict(self) -> dict[str, object]:
        return {
            "factor_columns": list(self.factor_columns),
            "raw_to_label": {str(k): v for k, v in self.raw_to_label.items()},
            "label_to_raw": self.label_to_raw,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "startprob": self.hmm.startprob_.tolist(),
            "transmat": self.hmm.transmat_.tolist(),
            "means": self.hmm.means_.tolist(),
            "covars": self.hmm.covars_.tolist(),
            "loglik_monitor": self.hmm.monitor_,
        }


def split_train_frame(factors: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    train_end = pd.to_datetime(config.train_end_date)
    train = factors.loc[factors["date"] <= train_end].copy()
    if len(train) < 250:
        cutoff = int(len(factors) * config.train_fraction_if_needed)
        train = factors.iloc[: max(cutoff, 250)].copy()
    return train


def name_states(hmm: DiagonalGaussianHMM, factor_columns: tuple[str, ...]) -> tuple[dict[int, str], pd.DataFrame]:
    means = pd.DataFrame(hmm.means_, columns=factor_columns)
    means["raw_state"] = range(hmm.n_states)
    high_state = int(means.sort_values(["E", "D"], ascending=[False, True]).iloc[0]["raw_state"])
    remaining = means.loc[means["raw_state"] != high_state].copy()
    remaining["bull_score"] = remaining["D"] + 0.5 * remaining["B"] + 0.25 * remaining["F"] - 0.25 * remaining["E"]
    bull_state = int(remaining.sort_values("bull_score", ascending=False).iloc[0]["raw_state"])
    bear_state = int(remaining.loc[remaining["raw_state"] != bull_state, "raw_state"].iloc[0])

    mapping = {high_state: "H", bull_state: "L+", bear_state: "L-"}
    means["label"] = means["raw_state"].map(mapping)
    ordered = means.sort_values("label").reset_index(drop=True)
    return mapping, ordered


def fit_state_model(factors: pd.DataFrame, config: MarketStateConfig) -> FittedMarketStateModel:
    factor_columns = config.state_factor_columns
    train = split_train_frame(factors, config)
    x_train = train.loc[:, factor_columns].to_numpy(dtype=float)
    hmm = DiagonalGaussianHMM(config.hmm_n_states, len(factor_columns), config).fit(x_train, factor_columns)
    raw_to_label, state_table = name_states(hmm, factor_columns)
    label_to_raw = {label: raw for raw, label in raw_to_label.items()}
    return FittedMarketStateModel(
        hmm=hmm,
        factor_columns=factor_columns,
        raw_to_label=raw_to_label,
        label_to_raw=label_to_raw,
        train_start=pd.to_datetime(train["date"].min()),
        train_end=pd.to_datetime(train["date"].max()),
        state_table=state_table,
    )
