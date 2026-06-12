from __future__ import annotations

import json
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
module_dir_text = str(MODULE_DIR)
if module_dir_text in sys.path:
    sys.path.remove(module_dir_text)
project_dir_text = str(PROJECT_DIR)
if project_dir_text not in sys.path:
    sys.path.insert(0, project_dir_text)

import numpy as np
import pandas as pd

from market_state_model.config import MarketStateConfig
from market_state_model.hmm_model import DiagonalGaussianHMM, split_train_frame


OUT_DIR = MODULE_DIR / "output" / "feature_tuned_direct_hmm"
FACTOR_PATH = MODULE_DIR / "output" / "intermediate" / "state_factors.csv"
STRATEGY_COMPARISON_PATH = (
    MODULE_DIR / "output" / "risk_on_stable_experiment" / "risk_on_stable_strategy_comparison.csv"
)
HIGH_GAP_DAILY = (
    PROJECT_DIR
    / "optimal_leverage_model"
    / "output"
    / "resset_high_gap_hedge_by_period"
    / "resset_high_gap_hedge_daily_returns.csv"
)

TRADING_DAYS = 252
TUNED_FEATURE_COLUMNS = (
    "E",
    "D_soft",
    "B",
    "RiskOn",
    "CalmRiskOn",
    "BearPressure",
    "FundingRiskOn",
)
PERIODS = {
    "full": (None, None),
    "2021": ("2021-01-01", "2021-12-31"),
    "2021_2023": ("2021-01-01", "2023-12-31"),
}
INDEX_NAME_MAP = {
    "000300": "沪深300",
    "000510": "中证A500",
    "000852": "中证1000",
    "000905": "中证500",
    "000906": "中证800",
    "000985": "中证全指",
    "932000": "中证2000",
}


class SeededDiagonalGaussianHMM(DiagonalGaussianHMM):
    def __init__(
        self,
        n_states: int,
        n_features: int,
        config: MarketStateConfig,
        initial_labels: np.ndarray,
    ):
        super().__init__(n_states, n_features, config)
        self.initial_labels = np.asarray(initial_labels, dtype=int)

    def _initial_labels(self, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        if len(self.initial_labels) != len(x):
            raise ValueError("Initial labels length does not match training sample length.")
        return self.initial_labels.copy()


def add_tuned_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["D_soft"] = 0.55 * out["D"]
    out["RiskOn"] = (
        0.30 * out["D"]
        + 0.35 * out["B"]
        + 0.15 * out["Liq"]
        + 0.25 * out["F"]
        - 0.20 * out["E"]
    )
    out["CalmRiskOn"] = (
        0.20 * out["D"]
        + 0.45 * out["B"]
        + 0.20 * out["Liq"]
        + 0.35 * out["F"]
        - 0.65 * out["E"]
        - 0.20 * out["D"].abs()
    )
    out["BearPressure"] = (
        -0.45 * out["D"]
        - 0.35 * out["B"]
        - 0.30 * out["F"]
        + 0.25 * out["E"]
    )
    out["FundingRiskOn"] = 0.50 * out["F"] + 0.25 * out["Liq"] + 0.25 * out["B"] - 0.20 * out["E"]
    return out


def lplus_score(frame: pd.DataFrame) -> pd.Series:
    return (
        0.45 * frame["RiskOn"]
        + 0.45 * frame["CalmRiskOn"]
        + 0.20 * frame["FundingRiskOn"]
        - 0.10 * frame["E"]
    )


def high_uncertainty_score(frame: pd.DataFrame) -> pd.Series:
    return frame["E"] + 0.25 * frame["BearPressure"] - 0.20 * frame["CalmRiskOn"]


def bear_score(frame: pd.DataFrame) -> pd.Series:
    return frame["BearPressure"] - 0.25 * frame["RiskOn"] - 0.15 * frame["CalmRiskOn"]


def stable_score(frame: pd.DataFrame) -> pd.Series:
    return (
        -0.35 * frame["E"]
        - 0.30 * frame["D_soft"].abs()
        - 0.25 * frame["RiskOn"].abs()
        - 0.25 * frame["BearPressure"].abs()
    )


def ensure_all_states(labels: np.ndarray, ranking: np.ndarray, n_states: int) -> np.ndarray:
    out = labels.copy()
    for state in range(n_states):
        if not np.any(out == state):
            out[int(ranking[state % len(ranking)])] = state
    return out


def candidate_labels(train: pd.DataFrame, n_states: int) -> list[tuple[str, np.ndarray]]:
    h = high_uncertainty_score(train).to_numpy(dtype=float)
    bull = lplus_score(train).to_numpy(dtype=float)
    bear = bear_score(train).to_numpy(dtype=float)
    stable = stable_score(train).to_numpy(dtype=float)
    ranking = np.argsort(h)[::-1]
    candidates: list[tuple[str, np.ndarray]] = []

    if n_states == 3:
        for h_q, bull_q in [(0.72, 0.55), (0.75, 0.55), (0.70, 0.60), (0.78, 0.58)]:
            labels = np.full(len(train), 2, dtype=int)
            labels[h >= np.nanquantile(h, h_q)] = 0
            labels[bull >= np.nanquantile(bull, bull_q)] = 1
            labels[(labels != 1) & (bear >= np.nanquantile(bear, 0.55))] = 2
            candidates.append((f"3s_h{int(h_q*100)}_bull{int(bull_q*100)}", ensure_all_states(labels, ranking, 3)))
    elif n_states == 4:
        for h_q, bull_q, bear_q in [(0.72, 0.58, 0.62), (0.75, 0.58, 0.60), (0.70, 0.60, 0.65)]:
            labels = np.full(len(train), 3, dtype=int)
            labels[h >= np.nanquantile(h, h_q)] = 0
            labels[bull >= np.nanquantile(bull, bull_q)] = 1
            labels[(labels != 1) & (bear >= np.nanquantile(bear, bear_q))] = 2
            labels[(labels == 3) & (stable < np.nanmedian(stable[labels == 3]))] = 2
            candidates.append(
                (
                    f"4s_h{int(h_q*100)}_bull{int(bull_q*100)}_bear{int(bear_q*100)}",
                    ensure_all_states(labels, ranking, 4),
                )
            )
    else:
        raise ValueError("Only 3-state and 4-state feature-tuned HMMs are supported.")

    return candidates


def fit_best_model(
    train: pd.DataFrame,
    config: MarketStateConfig,
    n_states: int,
) -> tuple[DiagonalGaussianHMM, str, pd.DataFrame]:
    x_train = train.loc[:, TUNED_FEATURE_COLUMNS].to_numpy(dtype=float)
    rows = []
    best_model: DiagonalGaussianHMM | None = None
    best_name = ""
    best_loglik = -np.inf

    for name, labels in candidate_labels(train, n_states):
        model = SeededDiagonalGaussianHMM(n_states, len(TUNED_FEATURE_COLUMNS), config, labels).fit(
            x_train, TUNED_FEATURE_COLUMNS
        )
        loglik = float(model.monitor_[-1])
        rows.append({"initialization": name, "train_loglik": loglik, "iterations": len(model.monitor_)})
        if loglik > best_loglik:
            best_model = model
            best_name = name
            best_loglik = loglik

    if best_model is None:
        raise RuntimeError(f"No {n_states}-state model was fitted.")
    return best_model, best_name, pd.DataFrame(rows).sort_values("train_loglik", ascending=False)


def name_states(
    hmm: DiagonalGaussianHMM,
    n_states: int,
) -> tuple[dict[int, str], pd.DataFrame]:
    means = pd.DataFrame(hmm.means_, columns=TUNED_FEATURE_COLUMNS)
    means["raw_state"] = range(hmm.n_states)
    means["h_score"] = high_uncertainty_score(means)
    means["lplus_score"] = lplus_score(means)
    means["bear_score"] = bear_score(means)
    means["stable_score"] = stable_score(means)

    high_state = int(means.sort_values(["h_score", "lplus_score"], ascending=[False, True]).iloc[0]["raw_state"])
    remaining = means.loc[means["raw_state"] != high_state].copy()
    bull_state = int(remaining.sort_values("lplus_score", ascending=False).iloc[0]["raw_state"])
    remaining = remaining.loc[remaining["raw_state"] != bull_state].copy()

    if n_states == 3:
        bear_state = int(remaining.iloc[0]["raw_state"])
        mapping = {high_state: "H", bull_state: "L+", bear_state: "L-"}
    else:
        bear_state = int(remaining.sort_values("bear_score", ascending=False).iloc[0]["raw_state"])
        stable_state = int(remaining.loc[remaining["raw_state"] != bear_state, "raw_state"].iloc[0])
        mapping = {high_state: "H", bull_state: "L+", stable_state: "Stable", bear_state: "L-"}

    means["label"] = means["raw_state"].map(mapping)
    return mapping, means.sort_values("label").reset_index(drop=True)


def forward_filter(
    hmm: DiagonalGaussianHMM,
    factors: pd.DataFrame,
    raw_to_label: dict[int, str],
) -> pd.DataFrame:
    x = factors.loc[:, TUNED_FEATURE_COLUMNS].to_numpy(dtype=float)
    b = hmm.emission_prob(x)
    a = hmm.transmat_
    start = hmm.startprob_
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

    label_to_raw = {label: raw for raw, label in raw_to_label.items()}
    raw_regime = np.argmax(alpha, axis=1)
    regimes = [raw_to_label[int(raw)] for raw in raw_regime]
    h = label_to_raw["H"]
    bull = label_to_raw["L+"]
    bear = label_to_raw["L-"]

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
    if "Stable" in label_to_raw:
        stable = label_to_raw["Stable"]
        out["p_stable"] = alpha[:, stable]
        out["xi_H_to_stable"] = xi[:, h, stable]
    return out


def add_signals(prob: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    out = prob.copy().sort_values("date").reset_index(drop=True)
    out["delta_p_low_bull"] = out["p_low_bull"].diff().fillna(0.0)
    out["delta_p_high_entropy"] = out["p_high_entropy"].diff().fillna(0.0)
    out["delta_p_low_bear"] = out["p_low_bear"].diff().fillna(0.0)
    if "p_stable" in out.columns:
        out["delta_p_stable"] = out["p_stable"].diff().fillna(0.0)
    out["transition_score"] = out["xi_H_to_Lplus"] - out["xi_H_to_Lminus"]
    out["entry_score"] = (
        out["p_low_bull"]
        + config.entry_gamma * out["delta_p_low_bull"]
        - config.entry_rho * out["p_low_bear"]
        - config.entry_xi * out["p_high_entropy"]
        - config.entry_omega * out["posterior_entropy"]
    )
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


def build_final_frame(scored: pd.DataFrame, factors: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        "date",
        "market_ret",
        "index_level",
        "realized_vol_20",
        "downside_vol_20",
        "cs_return_std",
        "advancer_ratio",
        "above_ma20_ratio",
        "above_ma60_ratio",
        "new_high_60_ratio",
        "new_low_60_ratio",
        "market_amount",
        "market_turnover_proxy",
        "margin_balance",
        "margin_net_buy_ratio",
        "data_quality_flags",
        "E",
        "D",
        "B",
        "Liq",
        "F",
        *TUNED_FEATURE_COLUMNS,
    ]
    merge_cols = list(dict.fromkeys(merge_cols))
    final = scored.merge(factors[merge_cols], on="date", how="left")
    final["data_split"] = "test"
    train_start = pd.to_datetime(train["date"].min())
    train_end = pd.to_datetime(train["date"].max())
    final.loc[(final["date"] >= train_start) & (final["date"] <= train_end), "data_split"] = "train"
    return final.sort_values("date").reset_index(drop=True)


def cumulative_return(ret: pd.Series) -> float:
    return float((1.0 + pd.to_numeric(ret, errors="coerce").fillna(0.0)).prod() - 1.0)


def max_drawdown(ret: pd.Series) -> float:
    wealth = (1.0 + pd.to_numeric(ret, errors="coerce").fillna(0.0)).cumprod()
    if wealth.empty:
        return np.nan
    return float((wealth / wealth.cummax() - 1.0).min())


def action_from_regime(regime: pd.Series) -> pd.Series:
    action = pd.Series("hold_index", index=regime.index, dtype="object")
    action.loc[regime.eq("L+")] = "buy_high_gap"
    action.loc[regime.isin(["H", "L-"])] = "remove_high_gap"
    return action


def strategy_metrics(final: pd.DataFrame, rule: str) -> pd.DataFrame:
    if not HIGH_GAP_DAILY.exists():
        return pd.DataFrame()
    daily = pd.read_csv(HIGH_GAP_DAILY, parse_dates=["trade_date"])
    daily["index_code"] = daily["index_code"].astype(str).str.zfill(6)
    daily = daily.loc[daily["selection_method"].eq("industry_neutral")].copy()

    signals = final[["date", "market_regime"]].rename(columns={"date": "trade_date"}).copy()
    signals["action"] = action_from_regime(signals["market_regime"]).shift(1).fillna("hold_index")
    merged = daily.merge(signals[["trade_date", "action"]], on="trade_date", how="left")
    merged["action"] = merged["action"].fillna("hold_index")

    ret = merged["base_return_resset"].copy()
    ret = ret.where(~merged["action"].eq("buy_high_gap"), merged["high_gap_return"])
    ret = ret.where(~merged["action"].eq("remove_high_gap"), merged["no_high_return"])
    merged["strategy_return"] = ret

    rows = []
    for index_code, index_frame in merged.groupby("index_code"):
        index_name = INDEX_NAME_MAP.get(index_code, str(index_frame["index_name"].iloc[0]))
        for period, (start, end) in PERIODS.items():
            frame = index_frame.copy()
            if start is not None:
                frame = frame.loc[frame["trade_date"] >= pd.to_datetime(start)]
            if end is not None:
                frame = frame.loc[frame["trade_date"] <= pd.to_datetime(end)]
            if frame.empty:
                continue
            active = frame["strategy_return"] - frame["base_return_resset"]
            active_std = active.std()
            rows.append(
                {
                    "index_code": index_code,
                    "index_name": index_name,
                    "rule": rule,
                    "period": period,
                    "n_days": len(frame),
                    "index_cum_return": cumulative_return(frame["base_return_resset"]),
                    "strategy_cum_return": cumulative_return(frame["strategy_return"]),
                    "excess_cum_vs_index": cumulative_return(frame["strategy_return"])
                    - cumulative_return(frame["base_return_resset"]),
                    "information_ratio": float(active.mean() / active_std * np.sqrt(TRADING_DAYS))
                    if active_std > 0
                    else np.nan,
                    "strategy_max_drawdown": max_drawdown(frame["strategy_return"]),
                    "buy_days": int(frame["action"].eq("buy_high_gap").sum()),
                    "remove_days": int(frame["action"].eq("remove_high_gap").sum()),
                    "hold_days": int(frame["action"].eq("hold_index").sum()),
                }
            )
    return pd.DataFrame(rows)


def read_baseline_csi800() -> pd.DataFrame:
    if not STRATEGY_COMPARISON_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(STRATEGY_COMPARISON_PATH)
    frame = frame.loc[
        frame["rule"].isin(["hmm4_original", "hmm4_stable_riskon_q60"])
        & frame["index_code"].astype(str).str.zfill(6).eq("000906")
    ].copy()
    frame["index_code"] = frame["index_code"].astype(str).str.zfill(6)
    frame["index_name"] = "中证800"
    return frame


def write_outputs(
    n_states: int,
    final: pd.DataFrame,
    model: DiagonalGaussianHMM,
    state_table: pd.DataFrame,
    init_table: pd.DataFrame,
    selected_init: str,
    raw_to_label: dict[int, str],
    train: pd.DataFrame,
    strategy_table: pd.DataFrame,
) -> None:
    subdir = OUT_DIR / f"{n_states}state"
    subdir.mkdir(parents=True, exist_ok=True)

    final.to_csv(subdir / f"market_state_probabilities_{n_states}state_feature_tuned.csv", index=False, encoding="utf-8-sig")
    transition_cols = [
        "date",
        "xi_H_to_Lplus",
        "xi_H_to_Lminus",
        "transition_score",
        "delta_p_low_bull",
        "delta_p_high_entropy",
        "delta_p_low_bear",
        "bullish_transition_signal",
        "bearish_transition_signal",
        "market_regime",
    ]
    if "xi_H_to_stable" in final.columns:
        transition_cols.insert(3, "xi_H_to_stable")
    if "delta_p_stable" in final.columns:
        transition_cols.insert(7, "delta_p_stable")
    final[transition_cols].to_csv(
        subdir / f"market_state_transitions_{n_states}state_feature_tuned.csv", index=False, encoding="utf-8-sig"
    )
    entry_cols = [
        "date",
        "entry_score",
        "p_low_bull",
        "p_low_bear",
        "p_high_entropy",
        "posterior_entropy",
        "market_regime",
    ]
    if "p_stable" in final.columns:
        entry_cols.insert(4, "p_stable")
    final[entry_cols].to_csv(subdir / f"entry_score_{n_states}state_feature_tuned.csv", index=False, encoding="utf-8-sig")
    init_table.to_csv(subdir / f"initialization_comparison_{n_states}state.csv", index=False, encoding="utf-8-sig")
    state_table.to_csv(subdir / f"state_table_{n_states}state.csv", index=False, encoding="utf-8-sig")
    if not strategy_table.empty:
        strategy_table.to_csv(subdir / f"strategy_comparison_{n_states}state.csv", index=False, encoding="utf-8-sig")

    params = {
        "n_states": n_states,
        "feature_columns": list(TUNED_FEATURE_COLUMNS),
        "feature_design": {
            "D_soft": "0.55*D",
            "RiskOn": "0.30*D + 0.35*B + 0.15*Liq + 0.25*F - 0.20*E",
            "CalmRiskOn": "0.20*D + 0.45*B + 0.20*Liq + 0.35*F - 0.65*E - 0.20*abs(D)",
            "BearPressure": "-0.45*D - 0.35*B - 0.30*F + 0.25*E",
            "FundingRiskOn": "0.50*F + 0.25*Liq + 0.25*B - 0.20*E",
        },
        "raw_to_label": {str(k): v for k, v in raw_to_label.items()},
        "label_to_raw": {label: raw for raw, label in raw_to_label.items()},
        "train_start": str(pd.to_datetime(train["date"].min()).date()),
        "train_end": str(pd.to_datetime(train["date"].max()).date()),
        "selected_initialization": selected_init,
        "startprob": model.startprob_.tolist(),
        "transmat": model.transmat_.tolist(),
        "means": model.means_.tolist(),
        "covars": model.covars_.tolist(),
        "loglik_monitor": model.monitor_,
        "uses_smoothed_probabilities": False,
        "uses_posthoc_absorption_or_state_aggregation": False,
    }
    (subdir / f"model_parameters_{n_states}state_feature_tuned.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无可用数据。_"
    return df.to_markdown(index=False, floatfmt=".4f")


def write_report(results: dict[int, dict[str, object]]) -> Path:
    lines = [
        "# 特征调节直出 HMM 实验报告",
        "",
        "## 实验目标",
        "",
        "本实验不做 `Stable -> L+` 的事后吸收，也不做五状态聚合。目标是通过重新选择 HMM 的观测特征，让模型自然直出三状态或四状态。",
        "",
        "三状态输出：`H / L+ / L-`。",
        "",
        "四状态输出：`H / L+ / Stable / L-`。",
        "",
        "其中 risk-on stable 不再作为一个单独标签，而是通过 `CalmRiskOn`、`RiskOn`、`FundingRiskOn` 等特征让它在特征空间上靠近 `L+`。",
        "",
        "## 调节后的观测特征",
        "",
        "- `E`：保留原不确定性因子。",
        "- `D_soft = 0.55*D`：弱化单纯方向动量，避免只有强趋势才被识别为 L+。",
        "- `RiskOn = 0.30*D + 0.35*B + 0.15*Liq + 0.25*F - 0.20*E`。",
        "- `CalmRiskOn = 0.20*D + 0.45*B + 0.20*Liq + 0.35*F - 0.65*E - 0.20*abs(D)`。",
        "- `BearPressure = -0.45*D - 0.35*B - 0.30*F + 0.25*E`。",
        "- `FundingRiskOn = 0.50*F + 0.25*Liq + 0.25*B - 0.20*E`。",
        "",
        "这些特征只由当期已有状态因子构造，不使用未来收益、不使用 Gap 表现、不使用平滑概率。",
    ]

    baseline = read_baseline_csi800()
    for n_states, payload in results.items():
        final = payload["final"]
        state_table = payload["state_table"]
        init_table = payload["init_table"]
        strategy = payload["strategy"]
        selected_init = payload["selected_init"]
        counts = final["market_regime"].value_counts().rename_axis("market_regime").reset_index(name="days")
        latest = final.iloc[-1]
        csi = strategy.loc[strategy["index_code"].astype(str).str.zfill(6).eq("000906")] if not strategy.empty else pd.DataFrame()
        compare = pd.concat([baseline, csi], ignore_index=True)
        keep_cols = [
            col
            for col in [
                "index_code",
                "index_name",
                "rule",
                "period",
                "n_days",
                "index_cum_return",
                "strategy_cum_return",
                "excess_cum_vs_index",
                "information_ratio",
                "strategy_max_drawdown",
                "buy_days",
                "remove_days",
                "hold_days",
            ]
            if col in compare.columns
        ]
        compare = compare[keep_cols] if not compare.empty else compare
        state_cols = [
            "raw_state",
            "label",
            "E",
            "D_soft",
            "B",
            "RiskOn",
            "CalmRiskOn",
            "BearPressure",
            "FundingRiskOn",
            "h_score",
            "lplus_score",
            "bear_score",
            "stable_score",
        ]
        lines.extend(
            [
                "",
                f"## {n_states} 状态直出结果",
                "",
                f"选中初始化：`{selected_init}`。",
                "",
                "### 初始化比较",
                "",
                table(init_table),
                "",
                "### 状态命名表",
                "",
                table(state_table[state_cols]),
                "",
                "### 状态分布",
                "",
                table(counts),
                "",
                "### 中证800策略对比",
                "",
                table(compare),
                "",
                "### 最新观测",
                "",
                f"- 日期：`{latest['date'].date()}`",
                f"- 状态：`{latest['market_regime']}`",
                f"- p_low_bull：`{latest['p_low_bull']:.4f}`",
                f"- p_high_entropy：`{latest['p_high_entropy']:.4f}`",
                f"- p_low_bear：`{latest['p_low_bear']:.4f}`",
                f"- entry_score：`{latest['entry_score']:.4f}`",
            ]
        )
        if "p_stable" in final.columns:
            lines.append(f"- p_stable：`{latest['p_stable']:.4f}`")

    lines.extend(
        [
            "",
            "## 结论口径",
            "",
            "如果要求完全避免复杂后处理，本实验的 3 状态和 4 状态输出是更干净的 HMM 直出规格。",
            "但是否替代原主模型，需要看后续 Gap 回归和稳健性检验，而不是只看某一年策略收益。",
        ]
    )
    report_path = OUT_DIR / "feature_tuned_direct_hmm_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path


def run_one(n_states: int, factors: pd.DataFrame, config: MarketStateConfig) -> dict[str, object]:
    train = split_train_frame(factors, config)
    model, selected_init, init_table = fit_best_model(train, config, n_states)
    raw_to_label, state_table = name_states(model, n_states)
    filtered = forward_filter(model, factors, raw_to_label)
    scored = add_signals(filtered, config)
    final = build_final_frame(scored, factors, train)
    strategy = strategy_metrics(final, f"feature_tuned_{n_states}state")
    write_outputs(n_states, final, model, state_table, init_table, selected_init, raw_to_label, train, strategy)
    return {
        "final": final,
        "model": model,
        "state_table": state_table,
        "init_table": init_table,
        "selected_init": selected_init,
        "strategy": strategy,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FACTOR_PATH.exists():
        raise FileNotFoundError(f"Missing factor file: {FACTOR_PATH}")

    config = MarketStateConfig()
    factors = pd.read_csv(FACTOR_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    factors = add_tuned_features(factors)

    results = {
        3: run_one(3, factors, config),
        4: run_one(4, factors, config),
    }
    report = write_report(results)
    print(report)
    for n_states in [3, 4]:
        print(OUT_DIR / f"{n_states}state" / f"market_state_probabilities_{n_states}state_feature_tuned.csv")


if __name__ == "__main__":
    main()
