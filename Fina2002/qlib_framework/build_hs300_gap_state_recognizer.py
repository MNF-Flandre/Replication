from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAP_DAILY = (
    PROJECT_ROOT
    / "qlib_framework"
    / "output"
    / "hs300_market_cap_gap_split"
    / "hs300_market_cap_gap_split_daily.csv"
)
DEFAULT_HMM = (
    PROJECT_ROOT
    / "market_state_model"
    / "output"
    / "feature_tuned_expanding_hmm_2015"
    / "4state"
    / "market_state_probabilities_4state_feature_tuned_expanding_quarterly.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "qlib_framework" / "output" / "hs300_gap_state_recognizer"

TRADING_DAYS = 252
COST_RATE = 0.0005
MIN_TRAIN_DAYS = 504
HORIZONS = [20, 60]
THRESHOLD_SPECS = [(0.65, 0.35), (0.70, 0.30)]

ACTION_RET_COL = {
    "index": "index_return",
    "high_gap": "high_gap_return",
    "no_high": "no_high_return",
}
TURNOVER = {
    ("index", "index"): 0.0,
    ("index", "high_gap"): 1.0,
    ("index", "no_high"): 0.4,
    ("high_gap", "index"): 1.0,
    ("high_gap", "high_gap"): 0.0,
    ("high_gap", "no_high"): 2.0,
    ("no_high", "index"): 0.4,
    ("no_high", "high_gap"): 2.0,
    ("no_high", "no_high"): 0.0,
}


RAW_MARKET_FEATURES = [
    "market_ret",
    "index_return",
    "index_return_5d",
    "index_return_20d",
    "index_return_60d",
    "index_vol_5d",
    "index_vol_20d",
    "index_vol_60d",
    "realized_vol_20",
    "downside_vol_20",
    "cs_return_std",
    "advancer_ratio",
    "advancer_ratio_mean5",
    "advancer_ratio_mean20",
    "above_ma20_ratio",
    "above_ma20_ratio_mean20",
    "above_ma60_ratio",
    "above_ma60_ratio_mean20",
    "new_high_60_ratio",
    "new_low_60_ratio",
    "market_turnover_proxy",
    "margin_net_buy_ratio",
    "posterior_entropy",
]

HMM_FEATURES = [
    "p_high_entropy",
    "p_high_entropy_mean20",
    "p_low_bull",
    "p_low_bull_mean20",
    "p_low_bear",
    "p_low_bear_mean20",
    "p_stable",
    "p_stable_mean20",
    "entry_score",
    "entry_score_mean20",
    "transition_score",
    "bullish_transition_signal",
    "bearish_transition_signal",
    "bullish_transition_past20",
    "bearish_transition_past20",
    "xi_H_to_Lplus",
    "xi_H_to_Lminus",
    "xi_H_to_stable",
    "delta_p_low_bull",
    "delta_p_high_entropy",
    "delta_p_low_bear",
    "delta_p_stable",
    "E",
    "D",
    "B",
    "Liq",
    "F",
    "D_soft",
    "RiskOn",
    "CalmRiskOn",
    "BearPressure",
    "FundingRiskOn",
    "regime_Lplus",
    "regime_Stable",
    "regime_H",
    "regime_Lminus",
]

HMM_FILTERED_ENGINEERED_FEATURES = [
    "posterior_entropy",
    "posterior_entropy_mean5",
    "posterior_entropy_mean20",
    "posterior_entropy_mean60",
    "posterior_entropy_std20",
    "p_high_entropy",
    "p_high_entropy_mean5",
    "p_high_entropy_mean20",
    "p_high_entropy_mean60",
    "p_high_entropy_std20",
    "p_high_entropy_change5",
    "p_high_entropy_change20",
    "p_low_bull",
    "p_low_bull_mean5",
    "p_low_bull_mean20",
    "p_low_bull_mean60",
    "p_low_bull_std20",
    "p_low_bull_change5",
    "p_low_bull_change20",
    "p_low_bear",
    "p_low_bear_mean5",
    "p_low_bear_mean20",
    "p_low_bear_mean60",
    "p_low_bear_std20",
    "p_low_bear_change5",
    "p_low_bear_change20",
    "p_stable",
    "p_stable_mean5",
    "p_stable_mean20",
    "p_stable_mean60",
    "p_stable_std20",
    "p_stable_change5",
    "p_stable_change20",
    "entry_score",
    "entry_score_mean5",
    "entry_score_mean20",
    "entry_score_mean60",
    "entry_score_std20",
    "entry_score_change5",
    "entry_score_change20",
    "transition_score",
    "transition_score_mean5",
    "transition_score_mean20",
    "transition_score_mean60",
    "transition_score_std20",
    "bullish_transition_signal",
    "bearish_transition_signal",
    "bullish_transition_past5",
    "bullish_transition_past20",
    "bullish_transition_past60",
    "bearish_transition_past5",
    "bearish_transition_past20",
    "bearish_transition_past60",
    "days_since_last_bullish_transition",
    "days_since_last_bearish_transition",
    "xi_H_to_Lplus",
    "xi_H_to_Lminus",
    "xi_H_to_stable",
    "delta_p_low_bull",
    "delta_p_high_entropy",
    "delta_p_low_bear",
    "delta_p_stable",
    "regime_Lplus",
    "regime_Stable",
    "regime_H",
    "regime_Lminus",
    "regime_changed",
    "regime_duration",
    "log_regime_duration",
    "regime_share20_Lplus",
    "regime_share20_Stable",
    "regime_share20_H",
    "regime_share20_Lminus",
    "regime_mode20_Lplus",
    "regime_mode20_Stable",
    "regime_mode20_H",
    "regime_mode20_Lminus",
    "regime_mode20_confidence",
]

HMM_OBSERVABLE_ENGINEERED_FEATURES = HMM_FILTERED_ENGINEERED_FEATURES + [
    "market_ret",
    "realized_vol_20",
    "downside_vol_20",
    "cs_return_std",
    "advancer_ratio",
    "advancer_ratio_mean5",
    "advancer_ratio_mean20",
    "above_ma20_ratio",
    "above_ma20_ratio_mean5",
    "above_ma20_ratio_mean20",
    "above_ma60_ratio",
    "above_ma60_ratio_mean5",
    "above_ma60_ratio_mean20",
    "new_high_60_ratio",
    "new_low_60_ratio",
    "market_turnover_proxy",
    "margin_net_buy_ratio",
    "E",
    "D",
    "B",
    "Liq",
    "F",
    "D_soft",
    "RiskOn",
    "CalmRiskOn",
    "BearPressure",
    "FundingRiskOn",
]

GAP_HISTORY_DIAGNOSTIC_FEATURES = [
    "high_vs_no_high_20d_lag1",
    "high_vs_no_high_60d_lag1",
    "high_vs_index_20d_lag1",
    "high_vs_index_60d_lag1",
]


@dataclass(frozen=True)
class ModelSpec:
    feature_set: str
    horizon: int
    upper_q: float
    lower_q: float
    mode: str = "spread"

    @property
    def key(self) -> str:
        if self.mode == "chooser":
            return f"chooser_{self.feature_set}_h{self.horizon}"
        return f"{self.feature_set}_h{self.horizon}_q{int(self.upper_q * 100)}_{int(self.lower_q * 100)}"


def setup_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_paths = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    fonts = []
    for path in font_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            fonts.append(font_manager.FontProperties(fname=str(path)).get_name())
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = [
        *fonts,
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    return plt


def cumulative_return(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def annualized_return(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() ** (TRADING_DAYS / len(values)) - 1.0)


def annualized_vol(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(active: pd.Series) -> float:
    values = pd.to_numeric(active, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    sd = values.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(values.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(ret: pd.Series) -> float:
    nav = (1.0 + pd.to_numeric(ret, errors="coerce").fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1.0).min())


def forward_relative_return(high: pd.Series, low: pd.Series, horizon: int) -> pd.Series:
    high_log = np.log1p(pd.to_numeric(high, errors="coerce"))
    low_log = np.log1p(pd.to_numeric(low, errors="coerce"))
    spread_log = high_log - low_log
    # Future window is t+1 through t+h; current day is not included.
    return np.expm1(spread_log.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1)))


def forward_return(ret: pd.Series, horizon: int) -> pd.Series:
    log_ret = np.log1p(pd.to_numeric(ret, errors="coerce"))
    return np.expm1(log_ret.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1)))


def forward_label_end_date(dates: pd.Series, horizon: int) -> pd.Series:
    return dates.shift(-horizon)


def rolling_return(ret: pd.Series, window: int) -> pd.Series:
    log_ret = np.log1p(pd.to_numeric(ret, errors="coerce"))
    return np.expm1(log_ret.rolling(window, min_periods=window).sum())


def rolling_any(signal: pd.Series, window: int) -> pd.Series:
    return signal.astype(float).rolling(window, min_periods=1).max().fillna(0.0)


def days_since_signal(signal: pd.Series) -> pd.Series:
    values = pd.to_numeric(signal, errors="coerce").fillna(0.0).to_numpy()
    out = np.full(len(values), np.nan)
    last_seen = None
    for idx, value in enumerate(values):
        if value > 0:
            last_seen = idx
        if last_seen is not None:
            out[idx] = idx - last_seen
    return pd.Series(out, index=signal.index)


def add_hmm_windows(hmm: pd.DataFrame) -> pd.DataFrame:
    hmm = hmm.sort_values("date").copy()
    window_cols = [
        "entry_score",
        "transition_score",
        "posterior_entropy",
        "p_low_bull",
        "p_low_bear",
        "p_stable",
        "p_high_entropy",
    ]
    for col in window_cols:
        if col in hmm.columns:
            values = pd.to_numeric(hmm[col], errors="coerce")
            hmm[f"{col}_mean5"] = values.rolling(5, min_periods=1).mean()
            hmm[f"{col}_mean20"] = values.rolling(20, min_periods=1).mean()
            hmm[f"{col}_mean60"] = values.rolling(60, min_periods=1).mean()
            hmm[f"{col}_std20"] = values.rolling(20, min_periods=2).std(ddof=1)
            hmm[f"{col}_change5"] = values - values.shift(5)
            hmm[f"{col}_change20"] = values - values.shift(20)
    for col in ["bullish_transition_signal", "bearish_transition_signal"]:
        hmm[col] = hmm[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
        short_name = col.replace("_signal", "")
        hmm[f"{short_name}_past5"] = rolling_any(hmm[col], 5)
        hmm[f"{short_name}_past20"] = rolling_any(hmm[col], 20)
        hmm[f"{short_name}_past60"] = rolling_any(hmm[col], 60)
        hmm[f"days_since_last_{short_name}"] = days_since_signal(hmm[col])
    hmm["bullish_transition_past20"] = rolling_any(hmm["bullish_transition_signal"], 20)
    hmm["bearish_transition_past20"] = rolling_any(hmm["bearish_transition_signal"], 20)
    regimes = hmm["market_regime"].astype("string").fillna("missing")
    regime_map = {
        "L+": "Lplus",
        "Stable": "Stable",
        "H": "H",
        "L-": "Lminus",
    }
    regime_dummy_cols = []
    for regime, label in regime_map.items():
        col = f"regime_{label}"
        hmm[col] = regimes.eq(regime).astype(float)
        regime_dummy_cols.append(col)
        hmm[f"regime_share20_{label}"] = hmm[col].rolling(20, min_periods=1).mean()
    hmm["regime_changed"] = regimes.ne(regimes.shift()).astype(float)
    hmm.loc[hmm.index[0], "regime_changed"] = 0.0
    regime_run_id = hmm["regime_changed"].cumsum()
    hmm["regime_duration"] = hmm.groupby(regime_run_id).cumcount() + 1
    hmm["log_regime_duration"] = np.log1p(hmm["regime_duration"])
    share_cols = [f"regime_share20_{label}" for label in regime_map.values()]
    share_values = hmm[share_cols].to_numpy(dtype=float)
    mode_idx = np.nanargmax(share_values, axis=1)
    hmm["regime_mode20_confidence"] = np.nanmax(share_values, axis=1)
    for idx, label in enumerate(regime_map.values()):
        hmm[f"regime_mode20_{label}"] = (mode_idx == idx).astype(float)
    return hmm


def load_feature_panel(gap_path: Path, hmm_path: Path) -> pd.DataFrame:
    gap = pd.read_csv(gap_path, dtype={"index_code": "string"})
    gap["trade_date"] = pd.to_datetime(gap["trade_date"], errors="coerce")
    gap["index_return"] = pd.to_numeric(gap["base_return_resset"], errors="coerce")
    for col in ["high_gap_return", "no_high_return", "index_return"]:
        gap[col] = pd.to_numeric(gap[col], errors="coerce")
    gap = gap.sort_values("trade_date").copy()
    for window in [5, 20, 60]:
        gap[f"index_return_{window}d"] = rolling_return(gap["index_return"], window)
        gap[f"index_vol_{window}d"] = gap["index_return"].rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)

    if "high_vs_no_high_20d" not in gap.columns:
        gap["high_vs_no_high_20d"] = rolling_return(gap["high_gap_return"], 20) - rolling_return(gap["no_high_return"], 20)
    for col in ["high_vs_no_high_20d", "high_vs_no_high_60d", "high_vs_index_20d", "high_vs_index_60d"]:
        if col in gap.columns:
            gap[f"{col}_lag1"] = pd.to_numeric(gap[col], errors="coerce").shift(1)

    for horizon in HORIZONS:
        gap[f"future_high_vs_no_high_{horizon}d"] = forward_relative_return(
            gap["high_gap_return"], gap["no_high_return"], horizon
        )
        gap[f"future_high_gap_return_{horizon}d"] = forward_return(gap["high_gap_return"], horizon)
        gap[f"future_no_high_return_{horizon}d"] = forward_return(gap["no_high_return"], horizon)
        gap[f"future_index_return_{horizon}d"] = forward_return(gap["index_return"], horizon)
        gap[f"future_label_end_date_{horizon}d"] = forward_label_end_date(gap["trade_date"], horizon)

    hmm = pd.read_csv(hmm_path)
    hmm["date"] = pd.to_datetime(hmm["date"], errors="coerce")
    hmm = add_hmm_windows(hmm)
    for col in ["advancer_ratio", "above_ma20_ratio", "above_ma60_ratio"]:
        if col in hmm.columns:
            hmm[f"{col}_mean5"] = pd.to_numeric(hmm[col], errors="coerce").rolling(5, min_periods=1).mean()
            hmm[f"{col}_mean20"] = pd.to_numeric(hmm[col], errors="coerce").rolling(20, min_periods=1).mean()

    keep = ["date", "market_regime"]
    candidate_keep = (
        RAW_MARKET_FEATURES
        + HMM_FEATURES
        + HMM_FILTERED_ENGINEERED_FEATURES
        + HMM_OBSERVABLE_ENGINEERED_FEATURES
    )
    for col in candidate_keep:
        if col in hmm.columns and col not in keep:
            keep.append(col)
    panel = gap.merge(hmm[keep], left_on="trade_date", right_on="date", how="left")
    panel["market_regime"] = panel["market_regime"].fillna("missing")
    return panel


def feature_columns(feature_set: str, panel: pd.DataFrame) -> list[str]:
    if feature_set == "market_raw":
        candidates = RAW_MARKET_FEATURES
    elif feature_set == "market_raw_hmm":
        candidates = RAW_MARKET_FEATURES + HMM_FEATURES
    elif feature_set == "hmm_filtered_engineered":
        candidates = HMM_FILTERED_ENGINEERED_FEATURES
    elif feature_set == "hmm_observable_engineered":
        candidates = HMM_OBSERVABLE_ENGINEERED_FEATURES
    elif feature_set == "market_plus_gap_history_diagnostic":
        candidates = RAW_MARKET_FEATURES + HMM_FEATURES + GAP_HISTORY_DIAGNOSTIC_FEATURES
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    cols = [col for col in candidates if col in panel.columns]
    if not cols:
        raise ValueError(f"No feature columns found for {feature_set}")
    return list(dict.fromkeys(cols))


def fit_model():
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=140,
            learning_rate=0.035,
            num_leaves=7,
            max_depth=3,
            min_child_samples=60,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=42,
        )


def prepare_matrix(
    df: pd.DataFrame, cols: list[str], medians: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    out = df[cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if medians is None:
        medians = out.median(numeric_only=True)
    return out.fillna(medians).fillna(0.0), medians


def monthly_blocks(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    months = pd.to_datetime(dates).dt.to_period("M")
    blocks = []
    for _, idx in pd.Series(np.arange(len(dates))).groupby(months):
        block_dates = dates.iloc[idx.to_numpy()]
        blocks.append((block_dates.min(), block_dates.max()))
    return blocks


def walk_forward_predict(panel: pd.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.mode == "chooser":
        return walk_forward_choose(panel, spec)

    cols = feature_columns(spec.feature_set, panel)
    target = f"future_high_vs_no_high_{spec.horizon}d"
    label_end = f"future_label_end_date_{spec.horizon}d"
    df = panel.sort_values("trade_date").copy()
    df["score"] = np.nan
    df["score_upper_threshold"] = np.nan
    df["score_lower_threshold"] = np.nan
    df["state"] = "untrained"
    df["train_rows"] = 0
    df["model_key"] = spec.key
    importances = []

    dates = df["trade_date"].dropna().reset_index(drop=True)
    blocks = monthly_blocks(dates)
    for block_start, block_end in blocks:
        train_mask = (
            (df["trade_date"] < block_start)
            & (pd.to_datetime(df[label_end]) < block_start)
            & df[target].notna()
        )
        train_idx = df.index[train_mask]
        if len(train_idx) < MIN_TRAIN_DAYS:
            continue
        test_mask = (df["trade_date"] >= block_start) & (df["trade_date"] <= block_end)
        test_idx = df.index[test_mask]
        if len(test_idx) == 0:
            continue
        model = fit_model()
        x_train, medians = prepare_matrix(df.loc[train_idx], cols)
        y_train = pd.to_numeric(df.loc[train_idx, target], errors="coerce").fillna(0.0)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        upper = float(np.quantile(train_pred, spec.upper_q))
        lower = float(np.quantile(train_pred, spec.lower_q))
        x_test, _ = prepare_matrix(df.loc[test_idx], cols, medians)
        pred = model.predict(x_test)
        state = np.select(
            [pred >= upper, pred <= lower],
            ["forecast_high_gap_on", "forecast_high_gap_off"],
            default="neutral_index",
        )
        df.loc[test_idx, "score"] = pred
        df.loc[test_idx, "score_upper_threshold"] = upper
        df.loc[test_idx, "score_lower_threshold"] = lower
        df.loc[test_idx, "state"] = state
        df.loc[test_idx, "train_rows"] = len(train_idx)
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame(
                {
                    "model_key": spec.key,
                    "feature_set": spec.feature_set,
                    "horizon": spec.horizon,
                    "block_start": block_start,
                    "feature": cols,
                    "importance": model.feature_importances_,
                }
            )
            importances.append(imp)

    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return df, imp_df


def walk_forward_choose(panel: pd.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = feature_columns(spec.feature_set, panel)
    targets = {
        "high_gap": f"future_high_gap_return_{spec.horizon}d",
        "no_high": f"future_no_high_return_{spec.horizon}d",
        "index": f"future_index_return_{spec.horizon}d",
    }
    label_end = f"future_label_end_date_{spec.horizon}d"
    df = panel.sort_values("trade_date").copy()
    df["score"] = np.nan
    df["score_upper_threshold"] = np.nan
    df["score_lower_threshold"] = np.nan
    df["pred_high_gap"] = np.nan
    df["pred_no_high"] = np.nan
    df["pred_index"] = np.nan
    df["state"] = "untrained"
    df["train_rows"] = 0
    df["model_key"] = spec.key
    importances = []

    dates = df["trade_date"].dropna().reset_index(drop=True)
    blocks = monthly_blocks(dates)
    for block_start, block_end in blocks:
        train_mask = (df["trade_date"] < block_start) & (pd.to_datetime(df[label_end]) < block_start)
        for target in targets.values():
            train_mask &= df[target].notna()
        train_idx = df.index[train_mask]
        if len(train_idx) < MIN_TRAIN_DAYS:
            continue
        test_mask = (df["trade_date"] >= block_start) & (df["trade_date"] <= block_end)
        test_idx = df.index[test_mask]
        if len(test_idx) == 0:
            continue

        x_train, medians = prepare_matrix(df.loc[train_idx], cols)
        x_test, _ = prepare_matrix(df.loc[test_idx], cols, medians)
        pred_map: dict[str, np.ndarray] = {}
        for leg, target in targets.items():
            model = fit_model()
            y_train = pd.to_numeric(df.loc[train_idx, target], errors="coerce").fillna(0.0)
            model.fit(x_train, y_train)
            pred_map[leg] = model.predict(x_test)
            if hasattr(model, "feature_importances_"):
                importances.append(
                    pd.DataFrame(
                        {
                            "model_key": spec.key,
                            "feature_set": spec.feature_set,
                            "horizon": spec.horizon,
                            "target_leg": leg,
                            "block_start": block_start,
                            "feature": cols,
                            "importance": model.feature_importances_,
                        }
                    )
                )

        pred_frame = pd.DataFrame(pred_map, index=test_idx)
        best_leg = pred_frame[["high_gap", "no_high", "index"]].idxmax(axis=1)
        state = best_leg.map(
            {
                "high_gap": "forecast_high_gap_on",
                "no_high": "forecast_high_gap_off",
                "index": "neutral_index",
            }
        )
        df.loc[test_idx, "pred_high_gap"] = pred_frame["high_gap"].to_numpy()
        df.loc[test_idx, "pred_no_high"] = pred_frame["no_high"].to_numpy()
        df.loc[test_idx, "pred_index"] = pred_frame["index"].to_numpy()
        df.loc[test_idx, "score"] = (pred_frame["high_gap"] - pred_frame["no_high"]).to_numpy()
        df.loc[test_idx, "state"] = state.to_numpy()
        df.loc[test_idx, "train_rows"] = len(train_idx)

    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return df, imp_df


def action_from_state(state: str) -> str:
    if state == "forecast_high_gap_on":
        return "high_gap"
    if state == "forecast_high_gap_off":
        return "no_high"
    return "index"


def add_strategy_returns(pred: pd.DataFrame, cost_rate: float = COST_RATE) -> pd.DataFrame:
    df = pred.sort_values("trade_date").copy()
    # Trade on next day using the signal known at the previous close.
    df["raw_signal_available"] = df["state"].ne("untrained")
    df["signal_state"] = df["state"].where(df["state"].ne("untrained")).shift(1).fillna("neutral_index")
    df["signal_available"] = df["raw_signal_available"].shift(1).fillna(False).astype(bool)
    df["action"] = df["signal_state"].map(action_from_state).fillna("index")
    gross = []
    for _, row in df.iterrows():
        gross.append(row[ACTION_RET_COL[row["action"]]])
    df["strategy_gross_return"] = pd.to_numeric(pd.Series(gross, index=df.index), errors="coerce").fillna(0.0)
    df["prev_action"] = df["action"].shift(1).fillna("index")
    df["turnover"] = [
        TURNOVER.get((prev, cur), 1.0) for prev, cur in zip(df["prev_action"], df["action"])
    ]
    df["cost"] = df["turnover"] * cost_rate
    df["strategy_net_return"] = df["strategy_gross_return"] - df["cost"]
    df["active_net_return"] = df["strategy_net_return"] - df["index_return"]
    return df


def build_hmm_baseline(panel: pd.DataFrame, cost_rate: float = COST_RATE) -> pd.DataFrame:
    df = panel.sort_values("trade_date").copy()
    regime_signal = df["market_regime"].shift(1).fillna("Stable")
    df["model_key"] = "hmm_rule_Lplus_high_H_Lminus_nohigh"
    df["score"] = np.nan
    df["state"] = np.select(
        [regime_signal.eq("L+"), regime_signal.isin(["H", "L-"])],
        ["forecast_high_gap_on", "forecast_high_gap_off"],
        default="neutral_index",
    )
    df["signal_available"] = True
    df["signal_state"] = df["state"]
    df["action"] = df["signal_state"].map(action_from_state).fillna("index")
    gross = []
    for _, row in df.iterrows():
        gross.append(row[ACTION_RET_COL[row["action"]]])
    df["strategy_gross_return"] = pd.to_numeric(pd.Series(gross, index=df.index), errors="coerce").fillna(0.0)
    df["prev_action"] = df["action"].shift(1).fillna("index")
    df["turnover"] = [
        TURNOVER.get((prev, cur), 1.0) for prev, cur in zip(df["prev_action"], df["action"])
    ]
    df["cost"] = df["turnover"] * cost_rate
    df["strategy_net_return"] = df["strategy_gross_return"] - df["cost"]
    df["active_net_return"] = df["strategy_net_return"] - df["index_return"]
    return df


def action_hit_stats(df: pd.DataFrame, horizon: int) -> dict[str, float | int]:
    target = f"future_high_vs_no_high_{horizon}d"
    valid = df[df[target].notna() & df["state"].ne("untrained")].copy()
    on = valid["state"].eq("forecast_high_gap_on")
    off = valid["state"].eq("forecast_high_gap_off")
    neutral = valid["state"].eq("neutral_index")
    out: dict[str, float | int] = {
        "n_signal_days": int(len(valid)),
        "on_days": int(on.sum()),
        "off_days": int(off.sum()),
        "neutral_days": int(neutral.sum()),
        "on_share": float(on.mean()) if len(valid) else np.nan,
        "off_share": float(off.mean()) if len(valid) else np.nan,
        "neutral_share": float(neutral.mean()) if len(valid) else np.nan,
    }
    out["on_future_spread_mean"] = float(valid.loc[on, target].mean()) if on.any() else np.nan
    out["off_future_spread_mean"] = float(valid.loc[off, target].mean()) if off.any() else np.nan
    out["neutral_future_spread_mean"] = float(valid.loc[neutral, target].mean()) if neutral.any() else np.nan
    out["on_hit_rate"] = float((valid.loc[on, target] > 0).mean()) if on.any() else np.nan
    out["off_hit_rate"] = float((valid.loc[off, target] < 0).mean()) if off.any() else np.nan
    try:
        out["score_ic"] = float(valid["score"].corr(valid[target], method="spearman"))
    except Exception:
        out["score_ic"] = np.nan
    return out


def summarize_strategy(
    df: pd.DataFrame,
    model_key: str,
    feature_set: str,
    horizon: int | str,
    eval_start: pd.Timestamp | None = None,
) -> dict[str, object]:
    valid = df[df["strategy_net_return"].notna()].copy()
    if "signal_available" in valid.columns:
        valid = valid[valid["signal_available"].astype(bool)].copy()
    if eval_start is not None:
        valid = valid[valid["trade_date"] >= eval_start].copy()
    if valid.empty:
        return {"model_key": model_key, "feature_set": feature_set, "horizon": horizon, "n_days": 0}
    active = valid["strategy_net_return"] - valid["index_return"]
    stats = {
        "model_key": model_key,
        "feature_set": feature_set,
        "horizon": horizon,
        "n_days": int(len(valid)),
        "start_date": valid["trade_date"].min().date().isoformat(),
        "end_date": valid["trade_date"].max().date().isoformat(),
        "strategy_net_return": cumulative_return(valid["strategy_net_return"]),
        "strategy_gross_return": cumulative_return(valid["strategy_gross_return"]),
        "index_return": cumulative_return(valid["index_return"]),
        "net_excess_vs_index": cumulative_return(valid["strategy_net_return"]) - cumulative_return(valid["index_return"]),
        "ann_return": annualized_return(valid["strategy_net_return"]),
        "ann_vol": annualized_vol(valid["strategy_net_return"]),
        "active_ir": information_ratio(active),
        "max_drawdown": max_drawdown(valid["strategy_net_return"]),
        "index_max_drawdown": max_drawdown(valid["index_return"]),
        "total_cost": float(valid["cost"].sum()),
        "avg_daily_turnover": float(valid["turnover"].mean()),
        "switch_days": int(valid["action"].ne(valid["action"].shift()).sum()),
        "high_gap_action_share": float(valid["action"].eq("high_gap").mean()),
        "no_high_action_share": float(valid["action"].eq("no_high").mean()),
        "index_action_share": float(valid["action"].eq("index").mean()),
    }
    if isinstance(horizon, int):
        stats.update(action_hit_stats(valid, horizon))
    return stats


def build_annual_summary(
    df: pd.DataFrame, model_key: str, eval_start: pd.Timestamp | None = None
) -> pd.DataFrame:
    rows = []
    tmp = df.copy()
    if "signal_available" in tmp.columns:
        tmp = tmp[tmp["signal_available"].astype(bool)].copy()
    if eval_start is not None:
        tmp = tmp[tmp["trade_date"] >= eval_start].copy()
    tmp["year"] = tmp["trade_date"].dt.year
    for year, part in tmp.groupby("year", sort=True):
        rows.append(
            {
                "model_key": model_key,
                "year": int(year),
                "strategy_net_return": cumulative_return(part["strategy_net_return"]),
                "index_return": cumulative_return(part["index_return"]),
                "net_excess_vs_index": cumulative_return(part["strategy_net_return"]) - cumulative_return(part["index_return"]),
                "active_ir": information_ratio(part["strategy_net_return"] - part["index_return"]),
                "total_cost": float(part["cost"].sum()),
                "avg_daily_turnover": float(part["turnover"].mean()),
                "high_gap_action_share": float(part["action"].eq("high_gap").mean()),
                "no_high_action_share": float(part["action"].eq("no_high").mean()),
                "index_action_share": float(part["action"].eq("index").mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_nav(comparison: pd.DataFrame, output: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.plot(comparison["trade_date"], comparison["index_nav"], label="沪深300指数", linewidth=2.1)
    ax.plot(comparison["trade_date"], comparison["hmm_nav"], label="HMM规则", linewidth=1.8)
    ax.plot(comparison["trade_date"], comparison["best_nav"], label="高Gap专用状态识别器", linewidth=2.2)
    ax.set_title("沪深300：高Gap专用状态识别器 vs HMM")
    ax.set_ylabel("净值，起点=1")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_active(comparison: pd.DataFrame, output: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.plot(
        comparison["trade_date"],
        comparison["best_nav"] / comparison["index_nav"] - 1.0,
        label="识别器 - 指数",
        linewidth=2.1,
    )
    ax.plot(
        comparison["trade_date"],
        comparison["hmm_nav"] / comparison["index_nav"] - 1.0,
        label="HMM规则 - 指数",
        linewidth=1.8,
    )
    ax.set_title("主动净值：高Gap专用状态识别器 vs HMM")
    ax.set_ylabel("相对指数累计收益")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_state_timeline(best_daily: pd.DataFrame, output: Path) -> None:
    plt = setup_matplotlib()
    colors = {"high_gap": "#B33A3A", "no_high": "#2D6B9F", "index": "#6A8E3F"}
    action_num = best_daily["action"].map({"no_high": -1, "index": 0, "high_gap": 1}).fillna(0)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True, height_ratios=[2.2, 1.0])
    axes[0].plot(best_daily["trade_date"], best_daily["score"], color="#333333", linewidth=1.2)
    axes[0].plot(best_daily["trade_date"], best_daily["score_upper_threshold"], color="#B33A3A", linestyle="--", linewidth=0.9)
    axes[0].plot(best_daily["trade_date"], best_daily["score_lower_threshold"], color="#2D6B9F", linestyle="--", linewidth=0.9)
    axes[0].set_title("高Gap专用状态识别器：预测分数与动态阈值")
    axes[0].set_ylabel("预测未来相对收益")
    axes[0].grid(True, alpha=0.25)
    for action, color in colors.items():
        mask = best_daily["action"].eq(action)
        axes[1].scatter(best_daily.loc[mask, "trade_date"], action_num.loc[mask], s=8, color=color, label=action)
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["非高Gap", "指数", "高Gap"])
    axes[1].set_title("次日执行动作")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, output: Path) -> None:
    if importance.empty:
        return
    plt = setup_matplotlib()
    top = (
        importance.groupby("feature", as_index=False)["importance"].mean()
        .sort_values("importance", ascending=False)
        .head(18)
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(top["feature"], top["importance"], color="#2D6B9F")
    ax.set_title("高Gap专用状态识别器：平均特征重要性")
    ax.set_xlabel("LightGBM split importance")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    best_key: str,
    best_daily: pd.DataFrame,
    hmm_daily: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame, max_rows: int = 20, floatfmt: str = ".4f") -> str:
        if df.empty:
            return "_无数据_"
        try:
            return df.head(max_rows).to_markdown(index=False, floatfmt=floatfmt)
        except Exception:
            return df.head(max_rows).to_csv(index=False)

    best = summary.loc[summary["model_key"].eq(best_key)].iloc[0]
    eval_start = pd.to_datetime(best["start_date"])
    hmm = summarize_strategy(
        hmm_daily,
        "hmm_rule_Lplus_high_H_Lminus_nohigh",
        "hmm_feature_tuned_4state_rule",
        "hmm",
        eval_start=eval_start,
    )
    ranked = summary.sort_values(["net_excess_vs_index", "active_ir"], ascending=False)
    imp_top = (
        importance.groupby("feature", as_index=False)["importance"].mean()
        .sort_values("importance", ascending=False)
        .head(20)
        if not importance.empty
        else pd.DataFrame()
    )

    best_oos = best_daily[
        (best_daily["trade_date"] >= eval_start) & best_daily["signal_available"].astype(bool)
    ].copy()
    hmm_oos = hmm_daily[hmm_daily["trade_date"] >= eval_start].copy()
    action_profile = (
        best_oos.groupby(["signal_state", "action"], dropna=False)
        .agg(
            n_days=("trade_date", "size"),
            net_return=("strategy_net_return", lambda x: cumulative_return(x)),
            index_return=("index_return", lambda x: cumulative_return(x)),
            active_mean=("active_net_return", "mean"),
            turnover=("turnover", "mean"),
        )
        .reset_index()
    )
    hmm_action_profile = (
        hmm_oos.groupby(["state", "action"], dropna=False)
        .agg(
            n_days=("trade_date", "size"),
            net_return=("strategy_net_return", lambda x: cumulative_return(x)),
            index_return=("index_return", lambda x: cumulative_return(x)),
            active_mean=("active_net_return", "mean"),
            turnover=("turnover", "mean"),
        )
        .reset_index()
    )

    lines = [
        "# 沪深300高 Gap 专用市场状态识别器",
        "",
        "## 目标",
        "",
        "这个识别器不是重新训练 HMM，也不是修改 `d*` 或 Gap，而是在已有沪深300高/非高 Gap 市值加权收益和市场状态特征上，训练一个监督式 conditioning layer。",
        "",
        "目标标签是未来高 Gap 相对非高 Gap 的收益：",
        "",
        "- `future_high_vs_no_high_20d`",
        "- `future_high_vs_no_high_60d`",
        "",
        "预测分数高时，次日持有高 Gap；预测分数低时，次日持有非高 Gap；中性时持有指数。",
        "",
        "## 防泄漏设置",
        "",
        "- 每个月只用该月开始前已经完成标签窗口的数据训练。",
        "- 训练样本要求 `label_end_date < prediction_month_start`，因此不会把预测月及之后的高 Gap 表现放入训练。",
        "- 当日收盘后识别的状态，只在下一交易日执行。",
        "- 主模型不使用高 Gap 历史相对收益；`market_plus_gap_history_diagnostic` 只作为诊断，避免把状态识别器变成动量器。",
        "- 未使用 smoothed HMM probability。",
        "",
        "## 最优 walk-forward 结果",
        "",
        f"- 最优非诊断模型：`{best_key}`。",
        f"- 净收益：{best['strategy_net_return']:.2%}；指数：{best['index_return']:.2%}；相对指数：{best['net_excess_vs_index']:.2%}。",
        f"- 主动 IR：{best['active_ir']:.3f}；最大回撤：{best['max_drawdown']:.2%}；总交易成本：{best['total_cost']:.2%}。",
        f"- 同期 HMM 规则相对指数：{hmm['net_excess_vs_index']:.2%}；主动 IR：{hmm['active_ir']:.3f}。",
        "",
        "## 模型排名",
        "",
        table(
            ranked[
                [
                    "model_key",
                    "feature_set",
                    "horizon",
                    "strategy_net_return",
                    "index_return",
                    "net_excess_vs_index",
                    "active_ir",
                    "max_drawdown",
                    "total_cost",
                    "avg_daily_turnover",
                    "high_gap_action_share",
                    "no_high_action_share",
                    "index_action_share",
                    "score_ic",
                    "on_future_spread_mean",
                    "off_future_spread_mean",
                ]
            ],
            max_rows=20,
            floatfmt=".4f",
        ),
        "",
        "## 年度表现",
        "",
        table(
            annual[annual["model_key"].isin([best_key, "hmm_rule_Lplus_high_H_Lminus_nohigh"])][
                [
                    "model_key",
                    "year",
                    "strategy_net_return",
                    "index_return",
                    "net_excess_vs_index",
                    "active_ir",
                    "total_cost",
                    "high_gap_action_share",
                    "no_high_action_share",
                    "index_action_share",
                ]
            ].sort_values(["year", "model_key"]),
            max_rows=30,
            floatfmt=".4f",
        ),
        "",
        "## 最优识别器动作画像",
        "",
        table(action_profile, max_rows=20, floatfmt=".4f"),
        "",
        "## HMM 规则动作画像",
        "",
        table(hmm_action_profile, max_rows=20, floatfmt=".4f"),
        "",
        "## 特征重要性",
        "",
        table(imp_top, max_rows=20, floatfmt=".4f"),
        "",
        "## 解释",
        "",
        "- 如果最优模型来自 `market_raw`，说明仅靠市场原始宽度、波动、趋势和资金环境就能比 HMM 更贴合高 Gap。",
        "- 如果最优模型来自 `market_raw_hmm`，说明 HMM 概率仍有信息，但需要和更多市场原始变量组合后才能更好识别高 Gap 时机。",
        "- 这个识别器是为沪深300高 Gap 特化的，不能直接推广到中证500/1000，除非重新做 walk-forward 检验。",
        "- 结果仍是研究原型，不应称为完美识别器；真正进论文或策略前，需要扩展到不同指数、不同高 Gap 阈值、不同交易成本和更长 OOS。",
        "",
        "## 图表",
        "",
        "- `figures/hs300_gap_state_recognizer_nav.png`",
        "- `figures/hs300_gap_state_recognizer_active.png`",
        "- `figures/hs300_gap_state_recognizer_timeline.png`",
        "- `figures/hs300_gap_state_recognizer_feature_importance.png`",
    ]
    (output_dir / "hs300_gap_state_recognizer_report.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a supervised market-state recognizer tailored to HS300 High-Gap timing."
    )
    parser.add_argument("--gap-daily", default=str(DEFAULT_GAP_DAILY))
    parser.add_argument("--hmm", default=str(DEFAULT_HMM))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cost-rate", type=float, default=COST_RATE)
    parser.add_argument(
        "--feature-sets",
        default="market_raw,market_raw_hmm,market_plus_gap_history_diagnostic",
        help="Comma-separated feature sets to evaluate.",
    )
    parser.add_argument(
        "--chooser-only",
        action="store_true",
        help="Run only the three-leg chooser models and skip spread-threshold models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel = load_feature_panel(Path(args.gap_daily), Path(args.hmm))
    feature_sets = [item.strip() for item in str(args.feature_sets).split(",") if item.strip()]
    specs = []
    if not args.chooser_only:
        specs.extend(
            [
                ModelSpec(feature_set, horizon, upper, lower)
                for feature_set in feature_sets
                for horizon in HORIZONS
                for upper, lower in THRESHOLD_SPECS
            ]
        )
    specs.extend(
        [
            ModelSpec(feature_set, horizon, 0.0, 0.0, mode="chooser")
            for feature_set in feature_sets
            for horizon in HORIZONS
        ]
    )

    daily_parts = []
    importances = []
    for spec in specs:
        pred, imp = walk_forward_predict(panel, spec)
        strat = add_strategy_returns(pred, cost_rate=float(args.cost_rate))
        strat["model_key"] = spec.key
        strat["feature_set"] = spec.feature_set
        strat["horizon"] = spec.horizon
        daily_parts.append(strat)
        if not imp.empty:
            importances.append(imp)

    hmm_daily = build_hmm_baseline(panel, cost_rate=float(args.cost_rate))
    daily_all = pd.concat(daily_parts, ignore_index=True)
    eval_start = daily_all.loc[daily_all["signal_available"], "trade_date"].min()

    summary_rows = []
    annual_parts = []
    for spec, strat in zip(specs, daily_parts):
        summary_rows.append(
            summarize_strategy(strat, spec.key, spec.feature_set, spec.horizon, eval_start=eval_start)
        )
        annual_parts.append(build_annual_summary(strat, spec.key, eval_start=eval_start))

    summary_rows.append(
        summarize_strategy(
            hmm_daily,
            "hmm_rule_Lplus_high_H_Lminus_nohigh",
            "hmm_feature_tuned_4state_rule",
            "hmm",
            eval_start=eval_start,
        )
    )
    annual_parts.append(
        build_annual_summary(
            hmm_daily, "hmm_rule_Lplus_high_H_Lminus_nohigh", eval_start=eval_start
        )
    )

    importance_all = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values(["net_excess_vs_index", "active_ir"], ascending=False)
    annual = pd.concat(annual_parts, ignore_index=True)

    clean = summary[
        ~summary["feature_set"].astype(str).eq("market_plus_gap_history_diagnostic")
        & ~summary["model_key"].eq("hmm_rule_Lplus_high_H_Lminus_nohigh")
    ].copy()
    best_key = clean.sort_values(["net_excess_vs_index", "active_ir"], ascending=False)["model_key"].iloc[0]
    best_daily = daily_all[daily_all["model_key"].eq(best_key)].sort_values("trade_date").copy()

    best_oos_mask = best_daily["signal_available"].astype(bool)
    best_eval_start = best_daily.loc[best_oos_mask, "trade_date"].min()
    compare = best_daily[best_oos_mask & (best_daily["trade_date"] >= best_eval_start)][
        ["trade_date", "strategy_net_return", "index_return", "action", "score", "score_upper_threshold", "score_lower_threshold"]
    ].rename(columns={"strategy_net_return": "best_net_return", "action": "best_action"})
    hmm_compare = hmm_daily[hmm_daily["trade_date"] >= best_eval_start][["trade_date", "strategy_net_return", "action"]].rename(
        columns={"strategy_net_return": "hmm_net_return", "action": "hmm_action"}
    )
    compare = compare.merge(hmm_compare, on="trade_date", how="inner")
    compare["index_nav"] = (1.0 + compare["index_return"].fillna(0.0)).cumprod()
    compare["best_nav"] = (1.0 + compare["best_net_return"].fillna(0.0)).cumprod()
    compare["hmm_nav"] = (1.0 + compare["hmm_net_return"].fillna(0.0)).cumprod()

    daily_all.to_csv(output_dir / "hs300_gap_state_recognizer_daily_all.csv", index=False, encoding="utf-8-sig")
    daily_all.to_parquet(output_dir / "hs300_gap_state_recognizer_daily_all.parquet", index=False)
    best_daily.to_csv(output_dir / "hs300_gap_state_recognizer_best_daily.csv", index=False, encoding="utf-8-sig")
    hmm_daily.to_csv(output_dir / "hs300_gap_state_recognizer_hmm_baseline_daily.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(output_dir / "hs300_gap_state_recognizer_comparison_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "hs300_gap_state_recognizer_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(output_dir / "hs300_gap_state_recognizer_annual.csv", index=False, encoding="utf-8-sig")
    if not importance_all.empty:
        importance_all.to_csv(output_dir / "hs300_gap_state_recognizer_feature_importance.csv", index=False, encoding="utf-8-sig")

    plot_nav(compare, fig_dir / "hs300_gap_state_recognizer_nav.png")
    plot_active(compare, fig_dir / "hs300_gap_state_recognizer_active.png")
    plot_state_timeline(best_daily, fig_dir / "hs300_gap_state_recognizer_timeline.png")
    plot_importance(
        importance_all[importance_all["model_key"].eq(best_key)].copy(),
        fig_dir / "hs300_gap_state_recognizer_feature_importance.png",
    )
    write_report(output_dir, summary, annual, best_key, best_daily, hmm_daily, importance_all[importance_all["model_key"].eq(best_key)].copy())

    print("HS300 Gap state recognizer completed.")
    print(f"Output dir: {output_dir}")
    print(f"Best clean model: {best_key}")
    print(
        summary[
            [
                "model_key",
                "feature_set",
                "horizon",
                "strategy_net_return",
                "index_return",
                "net_excess_vs_index",
                "active_ir",
                "total_cost",
                "avg_daily_turnover",
                "score_ic",
                "on_future_spread_mean",
                "off_future_spread_mean",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
