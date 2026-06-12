from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MarketStateConfig
from .factors import FACTOR_COMPONENTS
from .hmm_model import FittedMarketStateModel


def _md_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "_无可用数据_"
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def write_state_summary_report(
    output_dir: Path,
    model: FittedMarketStateModel,
    final_df: pd.DataFrame,
    config: MarketStateConfig,
) -> Path:
    path = output_dir / "state_summary_report.md"
    train_mask = (final_df["date"] >= model.train_start) & (final_df["date"] <= model.train_end)
    train_df = final_df.loc[train_mask].copy()

    factor_summary = (
        train_df.groupby("market_regime")[list(config.state_factor_columns)]
        .mean()
        .reset_index()
        .sort_values("market_regime")
    )
    raw_state_table = model.state_table.copy()

    ordered_labels = ["H", "L+", "L-"]
    mat = pd.DataFrame(index=ordered_labels, columns=ordered_labels, dtype=float)
    for i_label in ordered_labels:
        for j_label in ordered_labels:
            mat.loc[i_label, j_label] = model.hmm.transmat_[model.label_to_raw[i_label], model.label_to_raw[j_label]]
    mat = mat.reset_index().rename(columns={"index": "from_state"})

    lines = [
        "# 市场状态 HMM 状态摘要",
        "",
        "## 模型设定",
        "",
        f"- 状态数：{config.hmm_n_states}",
        "- 观测分布：对角协方差 Gaussian HMM",
        f"- 训练命名样本：{model.train_start.date()} 至 {model.train_end.date()}",
        "- 输出交易信号使用 forward filtering 概率；smoothed probability 仅用于事后诊断。",
        "",
        "## 状态命名依据",
        "",
        _md_table(raw_state_table[["raw_state", "label", *config.state_factor_columns]]),
        "",
        "命名规则只使用训练样本估计出的状态因子均值：E 最高者命名为 H；其余状态按 D、B、F 改善程度区分 L+ 与 L-。",
        "",
        "## 训练样本内状态因子均值",
        "",
        _md_table(factor_summary),
        "",
        "## 状态转移矩阵",
        "",
        _md_table(mat),
        "",
        "## 因子组件",
        "",
    ]
    for factor, components in FACTOR_COMPONENTS.items():
        lines.append(f"- {factor}: " + ", ".join(components))
    lines.extend(["", "本模块不修改 `optimal_leverage_model`、`d^*` 或 `Gap`。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _strategy_metrics(df: pd.DataFrame, config: MarketStateConfig, model: FittedMarketStateModel) -> pd.DataFrame:
    train_mask = (df["date"] >= model.train_start) & (df["date"] <= model.train_end)
    threshold = df.loc[train_mask, "entry_score"].quantile(config.entry_quantile_for_validation)
    position = (df["entry_score"] >= threshold).astype(float)
    fwd_ret = df["market_ret"].shift(-1)
    strategy_ret = position * fwd_ret
    always_ret = fwd_ret
    rows = []
    for name, ret, pos in [
        ("entry_score_top_quantile", strategy_ret, position),
        ("always_on_index", always_ret, pd.Series(1.0, index=df.index)),
    ]:
        rows.append(
            {
                "strategy": name,
                "threshold": threshold if name == "entry_score_top_quantile" else np.nan,
                "mean_daily_return": ret.mean(),
                "annualized_return": ret.mean() * config.annualization,
                "annualized_vol": ret.std() * np.sqrt(config.annualization),
                "max_drawdown": _max_drawdown(ret),
                "active_rate": pos.mean(),
                "turnover": pos.diff().abs().fillna(0.0).mean(),
            }
        )
    return pd.DataFrame(rows)


def _golden_cross_lead(df: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    fast = df["index_level"].rolling(config.golden_cross_fast_window, min_periods=config.golden_cross_fast_window).mean()
    slow = df["index_level"].rolling(config.golden_cross_slow_window, min_periods=config.golden_cross_slow_window).mean()
    cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    signal_idx = np.flatnonzero(df["bullish_transition_signal"].to_numpy())
    cross_idx = np.flatnonzero(cross.fillna(False).to_numpy())
    lead_days = []
    for idx in signal_idx:
        future = cross_idx[(cross_idx >= idx) & (cross_idx <= idx + config.golden_cross_lead_window)]
        if len(future):
            lead_days.append(int(future[0] - idx))
    return pd.DataFrame(
        [
            {
                "bullish_signal_count": len(signal_idx),
                "matched_next_golden_cross_count": len(lead_days),
                "matched_ratio": len(lead_days) / len(signal_idx) if len(signal_idx) else np.nan,
                "median_lead_trading_days": float(np.median(lead_days)) if lead_days else np.nan,
                "mean_lead_trading_days": float(np.mean(lead_days)) if lead_days else np.nan,
            }
        ]
    )


def write_validation_report(
    output_dir: Path,
    model: FittedMarketStateModel,
    final_df: pd.DataFrame,
    smoothed_df: pd.DataFrame,
    config: MarketStateConfig,
) -> Path:
    path = output_dir / "validation_report.md"
    df = final_df.copy().sort_values("date").reset_index(drop=True)
    df["fwd_ret_5"] = df["index_level"].shift(-5) / df["index_level"] - 1.0
    df["fwd_ret_20"] = df["index_level"].shift(-20) / df["index_level"] - 1.0
    df["future_breadth_change_20"] = df["above_ma20_ratio"].shift(-20) - df["above_ma20_ratio"]

    state_chars = (
        df.groupby("market_regime")
        .agg(
            obs=("date", "count"),
            realized_vol_20=("realized_vol_20", "mean"),
            downside_vol_20=("downside_vol_20", "mean"),
            cs_return_std=("cs_return_std", "mean"),
            advancer_ratio=("advancer_ratio", "mean"),
            above_ma20_ratio=("above_ma20_ratio", "mean"),
            market_amount=("market_amount", "mean"),
            market_turnover_proxy=("market_turnover_proxy", "mean"),
            margin_balance=("margin_balance", "mean"),
            margin_net_buy_ratio=("margin_net_buy_ratio", "mean"),
            market_ret=("market_ret", "mean"),
            fwd_ret_5=("fwd_ret_5", "mean"),
            fwd_ret_20=("fwd_ret_20", "mean"),
            future_breadth_change_20=("future_breadth_change_20", "mean"),
        )
        .reset_index()
    )

    signal_df = df.loc[df["bullish_transition_signal"]].copy()
    signal_perf = pd.DataFrame(
        [
            {
                "signal_count": int(len(signal_df)),
                "mean_fwd_ret_5": signal_df["fwd_ret_5"].mean(),
                "mean_fwd_ret_20": signal_df["fwd_ret_20"].mean(),
                "hit_rate_fwd_ret_20": (signal_df["fwd_ret_20"] > 0).mean() if len(signal_df) else np.nan,
                "mean_future_breadth_change_20": signal_df["future_breadth_change_20"].mean(),
            }
        ]
    )

    strategy = _strategy_metrics(df, config, model)
    lead = _golden_cross_lead(df, config)

    smooth_compare = df[["date", "p_high_entropy", "p_low_bull", "p_low_bear"]].merge(smoothed_df, on="date", how="left")
    smooth_stats = pd.DataFrame(
        [
            {
                "state": "H",
                "mean_abs_filter_minus_smooth": (smooth_compare["p_high_entropy"] - smooth_compare["smooth_p_high_entropy"]).abs().mean(),
                "max_abs_filter_minus_smooth": (smooth_compare["p_high_entropy"] - smooth_compare["smooth_p_high_entropy"]).abs().max(),
            },
            {
                "state": "L+",
                "mean_abs_filter_minus_smooth": (smooth_compare["p_low_bull"] - smooth_compare["smooth_p_low_bull"]).abs().mean(),
                "max_abs_filter_minus_smooth": (smooth_compare["p_low_bull"] - smooth_compare["smooth_p_low_bull"]).abs().max(),
            },
            {
                "state": "L-",
                "mean_abs_filter_minus_smooth": (smooth_compare["p_low_bear"] - smooth_compare["smooth_p_low_bear"]).abs().mean(),
                "max_abs_filter_minus_smooth": (smooth_compare["p_low_bear"] - smooth_compare["smooth_p_low_bear"]).abs().max(),
            },
        ]
    )

    lines = [
        "# 市场状态 HMM 验证报告",
        "",
        "## 1. 状态经济含义检查",
        "",
        _md_table(state_chars),
        "",
        "解释口径：H 应体现更高波动/截面分歧；L+ 应体现更好的方向和广度；L- 应体现更弱的后续收益或广度。",
        "",
        "## 2. H -> L+ 转换信号后续表现",
        "",
        _md_table(signal_perf),
        "",
        "## 3. 转换信号相对 20/60 均线金叉的提前性",
        "",
        _md_table(lead),
        "",
        "说明：`median_lead_trading_days` 是 bullish_transition_signal 到未来 20 个交易日内下一次 20/60 金叉的距离；数值越大表示信号越早。",
        "",
        "## 4. EntryScore 简单风险收益检查",
        "",
        _md_table(strategy),
        "",
        "`EntryScore` 不是概率。这里使用训练样本 80% 分位阈值形成简单择时暴露，并检查收益、回撤和换手。",
        "",
        "## 5. 过滤概率与平滑概率差异",
        "",
        _md_table(smooth_stats),
        "",
        "平滑概率使用了全样本 forward-backward 信息，只能用于事后诊断；交易和预测输出必须使用 `p_high_entropy`、`p_low_bull`、`p_low_bear` 这组三个 forward filtering 概率。",
        "",
        "## 6. 接口说明",
        "",
        "HMM 输出是市场层面条件变量，可后续按日期或报告期与企业层面 `Gap` 合并，例如 `Gap * EntryScore`、`Gap * P(L+)`、`Gap * P(H)`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_model_parameters(output_dir: Path, model: FittedMarketStateModel) -> Path:
    path = output_dir / "model_parameters.json"
    path.write_text(json.dumps(model.to_parameter_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
