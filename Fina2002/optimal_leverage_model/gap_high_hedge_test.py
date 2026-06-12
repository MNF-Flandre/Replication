from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gap_index_enhancement import (
    DEFAULT_ALPHAAGENT_DAILY_PATH,
    DEFAULT_ETF_WEIGHT_DIR,
    DEFAULT_GAP_PATH,
    DEFAULT_RAWDATA_ROOT,
    DEFAULT_REPORT_PATH,
    attach_latest_gap,
    load_stock_next_returns,
    parse_index_code_filter,
    read_gap_signals,
    read_index_weights,
)
from validate_optimal_leverage import load_industry_map, markdown_table


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "index_high_gap_hedge"
TRADING_DAYS = 252


@dataclass(frozen=True)
class HedgeConfig:
    high_quantile: float
    low_quantile: float
    hedge_notional: float
    max_signal_age_days: int
    min_signal_weight_coverage: float
    min_fresh_names: int
    min_industry_group_n: int


def prepare_panel(args: argparse.Namespace, config: HedgeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_code_filter = parse_index_code_filter(args.index_codes)
    weights, source_manifest = read_index_weights(Path(args.etf_weight_dir), index_code_filter)
    min_date = weights["weight_date"].min()
    max_date = weights["weight_date"].max() + pd.Timedelta(days=10)
    firm_ids = set(weights["firm_id"].astype(int).unique())

    stock_returns = load_stock_next_returns(Path(args.daily_returns_path), firm_ids, min_date, max_date)
    trading_dates = np.sort(stock_returns["trade_date"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    gap_signals = read_gap_signals(Path(args.gap_path), Path(args.report_path), trading_dates)
    industry = load_industry_map(Path(args.rawdata_root))

    panel = weights.merge(
        stock_returns,
        left_on=["firm_id", "weight_date"],
        right_on=["firm_id", "trade_date"],
        how="inner",
    )
    panel = panel.dropna(subset=["next_return_1d"]).copy()
    panel["raw_weight_sum"] = panel.groupby(["index_code", "weight_date"])["index_weight"].transform("sum")
    panel = panel[panel["raw_weight_sum"] > 0].copy()
    panel["base_weight"] = panel["index_weight"] / panel["raw_weight_sum"]
    panel = panel.merge(industry, on="firm_id", how="left")
    panel = attach_latest_gap(panel, gap_signals)
    panel["days_since_gap_available"] = (
        panel["weight_date"] - panel["available_trade_date"]
    ).dt.days
    panel["has_fresh_gap_signal"] = (
        panel["leverage_gap"].notna()
        & panel["days_since_gap_available"].notna()
        & (panel["days_since_gap_available"] >= 0)
        & (panel["days_since_gap_available"] <= config.max_signal_age_days)
    )
    panel["industry_section_code"] = panel["industry_section_code"].fillna("UNKNOWN").astype(str)
    return panel, source_manifest


def add_raw_index_groups(panel: pd.DataFrame, config: HedgeConfig) -> pd.DataFrame:
    out = panel.copy()
    out["high_gap_raw_index"] = False
    out["low_gap_raw_index"] = False
    valid = out["has_fresh_gap_signal"]
    group_cols = ["index_code", "weight_date"]
    pct_rank = out.loc[valid].groupby(group_cols, dropna=False)["leverage_gap"].rank(
        method="first", pct=True
    )
    counts = out.loc[valid].groupby(group_cols, dropna=False)["leverage_gap"].transform("count")
    high = valid.copy()
    high.loc[valid] = (pct_rank >= 1.0 - config.high_quantile) & (counts >= config.min_fresh_names)
    low = valid.copy()
    low.loc[valid] = (pct_rank <= config.low_quantile) & (counts >= config.min_fresh_names)
    out["high_gap_raw_index"] = high.fillna(False).astype(bool)
    out["low_gap_raw_index"] = low.fillna(False).astype(bool)
    return out


def add_industry_neutral_groups(panel: pd.DataFrame, config: HedgeConfig) -> pd.DataFrame:
    out = panel.copy()
    out["high_gap_industry_neutral"] = False
    out["low_gap_industry_neutral"] = False
    valid = out["has_fresh_gap_signal"]
    group_cols = ["index_code", "weight_date", "industry_section_code"]
    pct_rank = out.loc[valid].groupby(group_cols, dropna=False)["leverage_gap"].rank(
        method="first", pct=True
    )
    counts = out.loc[valid].groupby(group_cols, dropna=False)["leverage_gap"].transform("count")
    high = valid.copy()
    high.loc[valid] = (
        (pct_rank >= 1.0 - config.high_quantile)
        & (counts >= config.min_industry_group_n)
    )
    low = valid.copy()
    low.loc[valid] = (
        (pct_rank <= config.low_quantile)
        & (counts >= config.min_industry_group_n)
    )
    out["high_gap_industry_neutral"] = high.fillna(False).astype(bool)
    out["low_gap_industry_neutral"] = low.fillna(False).astype(bool)
    return out


def group_weighted_average(
    group: pd.DataFrame,
    mask_col: str,
    value_col: str = "next_return_1d",
) -> float:
    selected = group[group[mask_col]]
    if selected.empty:
        return np.nan
    weights = selected["base_weight"]
    denom = float(weights.sum())
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(np.sum(selected[value_col] * weights) / denom)


def compute_daily_tests(panel: pd.DataFrame, config: HedgeConfig, selection_method: str) -> pd.DataFrame:
    high_col = f"high_gap_{selection_method}"
    low_col = f"low_gap_{selection_method}"
    rows: list[dict[str, Any]] = []
    for (index_code, weight_date), group in panel.groupby(["index_code", "weight_date"], sort=True):
        base_return = float(np.sum(group["base_weight"] * group["next_return_1d"]))
        fresh = group["has_fresh_gap_signal"]
        signal_weight_coverage = float(group.loc[fresh, "base_weight"].sum())
        fresh_names = int(fresh.sum())
        high_names = int(group[high_col].sum())
        low_names = int(group[low_col].sum())
        high_weight = float(group.loc[group[high_col], "base_weight"].sum())
        low_weight = float(group.loc[group[low_col], "base_weight"].sum())
        high_return = group_weighted_average(group, high_col)
        low_return = group_weighted_average(group, low_col)
        rest = group[~group[high_col]]
        rest_weight = float(rest["base_weight"].sum())
        no_high_return = (
            float(np.sum(rest["base_weight"] * rest["next_return_1d"]) / rest_weight)
            if rest_weight > 0
            else np.nan
        )
        enough = (
            signal_weight_coverage >= config.min_signal_weight_coverage
            and fresh_names >= config.min_fresh_names
            and high_names > 0
            and high_weight > 0
            and np.isfinite(high_return)
        )
        if not enough:
            continue
        rows.append(
            {
                "index_code": index_code,
                "weight_date": weight_date,
                "selection_method": selection_method,
                "base_return": base_return,
                "high_gap_return": high_return,
                "low_gap_return": low_return,
                "no_high_return": no_high_return,
                "no_high_active_return": no_high_return - base_return
                if np.isfinite(no_high_return)
                else np.nan,
                "hedged_high_active_return": -config.hedge_notional * (high_return - base_return),
                "low_minus_high_return": low_return - high_return
                if np.isfinite(low_return)
                else np.nan,
                "signal_weight_coverage": signal_weight_coverage,
                "fresh_names": fresh_names,
                "high_names": high_names,
                "low_names": low_names,
                "high_weight": high_weight,
                "low_weight": low_weight,
                "hedge_notional": config.hedge_notional,
            }
        )
    return pd.DataFrame(rows)


def summarize_series(series: pd.Series) -> dict[str, float | int]:
    r = pd.to_numeric(series, errors="coerce").dropna()
    if r.empty:
        return {
            "n_days": 0,
            "mean_daily": np.nan,
            "ann_return_arithmetic": np.nan,
            "ann_vol": np.nan,
            "information_ratio": np.nan,
            "win_rate": np.nan,
            "cumulative_return": np.nan,
        }
    ann = float(r.mean() * TRADING_DAYS)
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan
    return {
        "n_days": int(len(r)),
        "mean_daily": float(r.mean()),
        "ann_return_arithmetic": ann,
        "ann_vol": vol,
        "information_ratio": ann / vol if vol and np.isfinite(vol) else np.nan,
        "win_rate": float((r > 0).mean()),
        "cumulative_return": float(np.prod(1.0 + r) - 1.0),
    }


def summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    return_cols = {
        "no_high_active_return": "remove_high_gap_and_redistribute",
        "hedged_high_active_return": "index_plus_short_high_gap_overlay",
        "low_minus_high_return": "low_gap_minus_high_gap_spread",
    }
    rows: list[dict[str, Any]] = []
    for (index_code, method), group in daily.groupby(["index_code", "selection_method"], sort=True):
        common = {
            "index_code": index_code,
            "selection_method": method,
            "start_date": group["weight_date"].min(),
            "end_date": group["weight_date"].max(),
            "avg_signal_weight_coverage": float(group["signal_weight_coverage"].mean()),
            "avg_fresh_names": float(group["fresh_names"].mean()),
            "avg_high_names": float(group["high_names"].mean()),
            "avg_low_names": float(group["low_names"].mean()),
            "avg_high_weight": float(group["high_weight"].mean()),
            "avg_low_weight": float(group["low_weight"].mean()),
            "hedge_notional": float(group["hedge_notional"].iloc[0]),
        }
        for col, strategy_name in return_cols.items():
            stats = summarize_series(group[col])
            rows.append({"strategy": strategy_name, **common, **stats})
    return pd.DataFrame(rows)


def summarize_overall(daily: pd.DataFrame) -> pd.DataFrame:
    return_cols = {
        "no_high_active_return": "remove_high_gap_and_redistribute",
        "hedged_high_active_return": "index_plus_short_high_gap_overlay",
        "low_minus_high_return": "low_gap_minus_high_gap_spread",
    }
    rows: list[dict[str, Any]] = []
    for method, group in daily.groupby("selection_method", sort=True):
        for col, strategy_name in return_cols.items():
            mean_by_day = (
                group.groupby("weight_date", as_index=False)[col]
                .mean()
                .rename(columns={col: "mean_return"})
            )
            stats = summarize_series(mean_by_day["mean_return"])
            rows.append(
                {
                    "selection_method": method,
                    "strategy": strategy_name,
                    "n_indices": int(group["index_code"].nunique()),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    source_manifest: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    config: HedgeConfig,
) -> None:
    best = summary.sort_values(
        ["information_ratio", "ann_return_arithmetic"],
        ascending=[False, False],
    ).head(20)
    remove_high = summary[summary["strategy"].eq("remove_high_gap_and_redistribute")].copy()
    overlay = summary[summary["strategy"].eq("index_plus_short_high_gap_overlay")].copy()
    spread = summary[summary["strategy"].eq("low_gap_minus_high_gap_spread")].copy()

    lines = [
        "# 高 Gap 对冲指数增强检验",
        "",
        "## 检验设计",
        "",
        "- 样本限制在指数成分股内部。",
        "- Gap 使用冻结主口径 `variant_A_clean_main_results.csv`。",
        "- 信号可用日使用财报公告日 `Annodt`，只在公告日后的第一个交易日及以后使用。",
        "- 每个指数每天使用当日已经可得的最新 Gap，并评价下一交易日收益。",
        "- `raw_index`：直接在指数成分股内部按 Gap 排序。",
        "- `industry_neutral`：在指数成分股内部再按行业分组，在行业内识别高 Gap 与低 Gap。",
        "",
        "## 三种组合定义",
        "",
        "- `remove_high_gap_and_redistribute`：把高 Gap 成分权重降到 0，并把权重按原基准权重重配给其余成分。这是最接近“把高 Gap 冲掉”的 long-only 版本。",
        "- `index_plus_short_high_gap_overlay`：持有原指数，同时用 `hedge_notional` 做空高 Gap 篮子、等额加回指数收益，用来检验高 Gap 是否相对指数跑输。",
        "- `low_gap_minus_high_gap_spread`：做多低 Gap 篮子、做空高 Gap 篮子，是纯信号价差，不是 long-only 指数增强。",
        "",
        "## 参数",
        "",
        f"- `high_quantile`: {config.high_quantile}",
        f"- `low_quantile`: {config.low_quantile}",
        f"- `hedge_notional`: {config.hedge_notional}",
        f"- `max_signal_age_days`: {config.max_signal_age_days}",
        f"- `min_signal_weight_coverage`: {config.min_signal_weight_coverage}",
        f"- `min_fresh_names`: {config.min_fresh_names}",
        f"- `min_industry_group_n`: {config.min_industry_group_n}",
        "",
        "## 权重文件来源",
        "",
        markdown_table(source_manifest, max_rows=20),
        "",
        "## 跨指数平均结果",
        "",
        markdown_table(overall, max_rows=20),
        "",
        "## 去掉高 Gap 后重配",
        "",
        markdown_table(remove_high.sort_values(["selection_method", "information_ratio"], ascending=[True, False]), max_rows=30),
        "",
        "## 指数加做空高 Gap overlay",
        "",
        markdown_table(overlay.sort_values(["selection_method", "information_ratio"], ascending=[True, False]), max_rows=30),
        "",
        "## 低 Gap 减高 Gap 价差",
        "",
        markdown_table(spread.sort_values(["selection_method", "information_ratio"], ascending=[True, False]), max_rows=30),
        "",
        "## 表现最好的组合",
        "",
        markdown_table(best, max_rows=20),
        "",
        "## 输出文件",
        "",
        "- `high_gap_hedge_daily_returns.csv`：指数-日期层面的每日测试收益。",
        "- `high_gap_hedge_summary.csv`：指数层面汇总。",
        "- `high_gap_hedge_overall.csv`：跨指数等权平均汇总。",
        "- `high_gap_hedge_report.md`：本报告。",
        "",
        "## 解释",
        "",
        "如果 `remove_high_gap_and_redistribute` 或 `index_plus_short_high_gap_overlay` 的主动收益为正，说明把高 Gap 暴露冲掉有帮助。"
        "如果为负，说明高 Gap 成分在该指数和样本期内相对跑赢，冲掉高 Gap 会损失收益。",
        "",
        "当前结果是毛收益，没有扣除交易成本、融券成本、冲击成本和换手限制。正式策略还需要加入这些约束。",
    ]
    (output_dir / "high_gap_hedge_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HedgeConfig(
        high_quantile=float(args.high_quantile),
        low_quantile=float(args.low_quantile),
        hedge_notional=float(args.hedge_notional),
        max_signal_age_days=int(args.max_signal_age_days),
        min_signal_weight_coverage=float(args.min_signal_weight_coverage),
        min_fresh_names=int(args.min_fresh_names),
        min_industry_group_n=int(args.min_industry_group_n),
    )

    panel, source_manifest = prepare_panel(args, config)
    source_manifest.to_csv(output_dir / "high_gap_hedge_source_manifest.csv", index=False, encoding="utf-8-sig")
    panel = add_raw_index_groups(panel, config)
    panel = add_industry_neutral_groups(panel, config)

    daily_parts = [
        compute_daily_tests(panel, config, "raw_index"),
        compute_daily_tests(panel, config, "industry_neutral"),
    ]
    daily = pd.concat(daily_parts, ignore_index=True)
    summary = summarize_daily(daily)
    overall = summarize_overall(daily)

    daily.to_csv(output_dir / "high_gap_hedge_daily_returns.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "high_gap_hedge_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "high_gap_hedge_overall.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, source_manifest, summary, overall, config)

    print("High Gap hedge test completed.")
    index_code_filter = parse_index_code_filter(args.index_codes)
    print(f"Index code filter: {sorted(index_code_filter) if index_code_filter else 'all'}")
    print(f"Index codes: {daily['index_code'].nunique()}")
    print(f"Daily rows: {len(daily)}")
    print("Overall:")
    print(overall.to_string(index=False))
    print(f"Summary: {output_dir / 'high_gap_hedge_summary.csv'}")
    print(f"Report: {output_dir / 'high_gap_hedge_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test hedging/removing high Gap inside index constituents.")
    parser.add_argument("--gap-path", default=str(DEFAULT_GAP_PATH))
    parser.add_argument("--etf-weight-dir", default=str(DEFAULT_ETF_WEIGHT_DIR))
    parser.add_argument("--rawdata-root", default=str(DEFAULT_RAWDATA_ROOT))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--daily-returns-path", default=str(DEFAULT_ALPHAAGENT_DAILY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--high-quantile", type=float, default=0.20)
    parser.add_argument("--low-quantile", type=float, default=0.20)
    parser.add_argument("--hedge-notional", type=float, default=0.25)
    parser.add_argument("--max-signal-age-days", type=int, default=540)
    parser.add_argument("--min-signal-weight-coverage", type=float, default=0.50)
    parser.add_argument("--min-fresh-names", type=int, default=30)
    parser.add_argument("--min-industry-group-n", type=int, default=5)
    parser.add_argument(
        "--index-codes",
        default="",
        help="Comma-separated normalized index codes to keep, e.g. 000906,000010.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
