from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from validate_optimal_leverage import load_industry_map, markdown_table


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAWDATA_ROOT = Path(os.environ.get("QUANT_RAWDATA_ROOT", PROJECT_ROOT / "external_data")).expanduser()
DEFAULT_RETURNS_PATH = DEFAULT_RAWDATA_ROOT / "output" / "intermediate" / "stock_daily_returns.parquet"
DEFAULT_REPORT_PATH = DEFAULT_RAWDATA_ROOT / "3tables" / "IAR_Rept.csv"
DEFAULT_MAIN_RESULT = DEFAULT_OUTPUT_DIR / "variant_A_clean_main_results.csv"


HORIZONS = [21, 63, 126]
TERCILE_LABELS = ["Low Gap", "Middle Gap", "High Gap"]
QUINTILE_LABELS = ["Q1 Low Gap", "Q2", "Q3", "Q4", "Q5 High Gap"]


def read_main_result(path: Path) -> pd.DataFrame:
    usecols = [
        "firm_id",
        "period_date",
        "period_type",
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "leverage_gap",
        "leverage_status",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["firm_id"] = pd.to_numeric(df["firm_id"], errors="coerce")
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df = df.dropna(subset=["firm_id", "period_date", "leverage_gap"]).copy()
    df["firm_id"] = df["firm_id"].astype(int)
    return df


def read_disclosure_dates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Stkcd", "Accper", "Annodt"], low_memory=False)
    df["firm_id"] = pd.to_numeric(df["Stkcd"], errors="coerce")
    df["period_date"] = pd.to_datetime(df["Accper"], errors="coerce")
    df["announcement_date"] = pd.to_datetime(df["Annodt"], errors="coerce")
    df = df.dropna(subset=["firm_id", "period_date", "announcement_date"]).copy()
    df["firm_id"] = df["firm_id"].astype(int)
    return (
        df.groupby(["firm_id", "period_date"], as_index=False)
        .agg(announcement_date=("announcement_date", "min"))
    )


def read_daily_returns(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["Stkcd", "CloseDate", "ret"])
    df["firm_id"] = pd.to_numeric(df["Stkcd"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df = df.dropna(subset=["firm_id", "trade_date", "ret"]).copy()
    df = df[np.isfinite(df["ret"]) & (df["ret"] > -1.0)].copy()
    df["firm_id"] = df["firm_id"].astype(int)
    df["log_ret"] = np.log1p(df["ret"])
    df = df.sort_values(["firm_id", "trade_date"])
    return df[["firm_id", "trade_date", "ret", "log_ret"]]


def attach_forward_returns(events: pd.DataFrame, daily: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = events.copy()
    out["signal_trade_date"] = pd.NaT
    out["available_return_obs"] = 0
    for horizon in horizons:
        out[f"ret_{horizon}d"] = np.nan
        out[f"end_date_{horizon}d"] = pd.NaT

    daily_groups = {
        int(fid): grp.reset_index(drop=True)
        for fid, grp in daily.groupby("firm_id", sort=False)
    }

    for firm_id, event_idx in out.groupby("firm_id").groups.items():
        grp = daily_groups.get(int(firm_id))
        if grp is None or grp.empty:
            continue
        dates = grp["trade_date"].to_numpy(dtype="datetime64[ns]")
        log_ret = grp["log_ret"].to_numpy(dtype=float)
        cum_log = np.cumsum(log_ret)
        ann_dates = out.loc[event_idx, "announcement_date"].to_numpy(dtype="datetime64[ns]")
        # Strictly use trading days after announcement_date. This avoids treating
        # Accper or same-day disclosure as tradable information.
        start_keys = ann_dates + np.timedelta64(1, "D")
        start_pos = np.searchsorted(dates, start_keys, side="left")
        valid_start = start_pos < len(dates)
        valid_indices = np.asarray(list(event_idx))[valid_start]
        valid_pos = start_pos[valid_start]
        if len(valid_indices) == 0:
            continue
        out.loc[valid_indices, "signal_trade_date"] = dates[valid_pos]
        out.loc[valid_indices, "available_return_obs"] = len(dates) - valid_pos
        start_cum_before = cum_log[valid_pos] - log_ret[valid_pos]
        for horizon in horizons:
            end_pos = valid_pos + horizon - 1
            enough = end_pos < len(dates)
            if not enough.any():
                continue
            idx_h = valid_indices[enough]
            ret_h = np.exp(cum_log[end_pos[enough]] - start_cum_before[enough]) - 1.0
            out.loc[idx_h, f"ret_{horizon}d"] = ret_h
            out.loc[idx_h, f"end_date_{horizon}d"] = dates[end_pos[enough]]
    return out


def rank_bins_within_group(s: pd.Series, labels: list[str], min_n: int) -> pd.Series:
    result = pd.Series(pd.NA, index=s.index, dtype="object")
    valid = s.notna()
    n = int(valid.sum())
    q = len(labels)
    if n < max(q, min_n):
        return result
    ranks = s.loc[valid].rank(method="first")
    bin_idx = np.ceil(ranks / n * q).astype(int).clip(1, q)
    result.loc[valid] = [labels[i - 1] for i in bin_idx]
    return result


def assign_gap_bins(events: pd.DataFrame, min_cluster_n: int) -> pd.DataFrame:
    out = events.copy()
    out["gap_tercile"] = pd.NA
    out["gap_quintile"] = pd.NA
    eligible = out["industry_matched"] & out["signal_year"].notna()
    cluster_cols = ["signal_year", "industry_section_code"]
    out.loc[eligible, "gap_tercile"] = (
        out.loc[eligible].groupby(cluster_cols, dropna=False)["leverage_gap"]
        .transform(lambda s: rank_bins_within_group(s, TERCILE_LABELS, min_cluster_n))
    )
    out.loc[eligible, "gap_quintile"] = (
        out.loc[eligible].groupby(cluster_cols, dropna=False)["leverage_gap"]
        .transform(lambda s: rank_bins_within_group(s, QUINTILE_LABELS, max(min_cluster_n, 25)))
    )
    return out


def bin_return_summary(events: pd.DataFrame, bin_col: str, horizons: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        ret_col = f"ret_{horizon}d"
        valid = events[events[bin_col].notna() & events[ret_col].notna()].copy()
        for bin_name, grp in valid.groupby(bin_col, sort=False):
            rows.append(
                {
                    "bin_scheme": bin_col,
                    "gap_bin": bin_name,
                    "horizon_days": horizon,
                    "n": int(len(grp)),
                    "mean_gap": float(grp["leverage_gap"].mean()),
                    "median_gap": float(grp["leverage_gap"].median()),
                    "mean_return": float(grp[ret_col].mean()),
                    "median_return": float(grp[ret_col].median()),
                    "std_return": float(grp[ret_col].std()),
                    "pct_positive_return": float((grp[ret_col] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def cluster_spread_test(events: pd.DataFrame, bin_col: str, low_label: str, high_label: str, horizons: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cluster_cols = ["signal_year", "industry_section_code", "industry_section_name"]
    for horizon in horizons:
        ret_col = f"ret_{horizon}d"
        valid = events[events[bin_col].isin([low_label, high_label]) & events[ret_col].notna()].copy()
        cluster_means = (
            valid.groupby([*cluster_cols, bin_col], dropna=False)[ret_col]
            .mean()
            .reset_index()
        )
        pivot = cluster_means.pivot_table(
            index=cluster_cols,
            columns=bin_col,
            values=ret_col,
            aggfunc="mean",
        ).reset_index()
        if low_label not in pivot.columns or high_label not in pivot.columns:
            spread = pd.Series(dtype=float)
        else:
            pivot["high_minus_low"] = pivot[high_label] - pivot[low_label]
            spread = pivot["high_minus_low"].dropna()
        n_clusters = int(spread.notna().sum())
        mean_spread = float(spread.mean()) if n_clusters else np.nan
        se = float(spread.std(ddof=1) / math.sqrt(n_clusters)) if n_clusters > 1 else np.nan
        rows.append(
            {
                "bin_scheme": bin_col,
                "horizon_days": horizon,
                "low_label": low_label,
                "high_label": high_label,
                "n_clusters": n_clusters,
                "mean_high_minus_low": mean_spread,
                "se_cluster_mean": se,
                "t_cluster_mean": mean_spread / se if se and np.isfinite(se) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def disclosure_quality_table(events: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "A_clean_main rows", "value": len(events)},
        {"metric": "missing announcement_date", "value": int(events["announcement_date"].isna().sum())},
        {"metric": "announcement_date before period_date", "value": int((events["announcement_date"] < events["period_date"]).sum())},
        {"metric": "valid disclosure rows", "value": int(events["valid_disclosure"].sum())},
        {"metric": "rows with signal_trade_date", "value": int(events["signal_trade_date"].notna().sum())},
        {"metric": "industry matched rows", "value": int(events["industry_matched"].sum()) if "industry_matched" in events.columns else 0},
        {"metric": "tercile assigned rows", "value": int(events["gap_tercile"].notna().sum())},
        {"metric": "quintile assigned rows", "value": int(events["gap_quintile"].notna().sum())},
    ]
    for horizon in HORIZONS:
        rows.append({"metric": f"rows with ret_{horizon}d", "value": int(events[f"ret_{horizon}d"].notna().sum())})
    return pd.DataFrame(rows)


def write_report(output_dir: Path, result_dir: Path, quality: pd.DataFrame, summary: pd.DataFrame, spread: pd.DataFrame) -> None:
    tercile_summary = summary[summary["bin_scheme"].eq("gap_tercile")].copy()
    tercile_spread = spread[spread["bin_scheme"].eq("gap_tercile")].copy()
    lines = [
        "# Gap 分档与披露后收益差异检验",
        "",
        "## 口径",
        "",
        "- 主信号来自 `variant_A_clean_main_results.csv`，不改变最优负债率主公式。",
        "- 会计信息截止日使用 `period_date`，但它不是可交易信号日。",
        "- 信息披露日使用 `IAR_Rept.csv` 的 `Annodt`。",
        "- 收益窗口从 `Annodt` 之后第一个可用交易日开始，避免把报告截止日到披露日前的收益错误归因于 Gap。",
        "- 行业聚类使用 `RESSET_CINFO_1.csv` 的证监会行业门类；Gap 分档在“披露年份 × 行业门类”内部完成。",
        "- 当前收益为披露后的个股原始复合收益，未做风险调整或市场调整。",
        "",
        "## 样本质量",
        "",
        markdown_table(quality),
        "",
        "## Gap 三分位收益",
        "",
        markdown_table(tercile_summary),
        "",
        "## High Gap - Low Gap 差值",
        "",
        "差值先在每个“披露年份 × 行业门类”聚类内计算，再对聚类差值做均值和 t 统计。",
        "",
        markdown_table(tercile_spread),
        "",
        "## 输出文件",
        "",
        f"- 事件收益面板：`{result_dir / 'gap_return_event_panel.csv'}`",
        f"- 分档收益表：`{result_dir / 'gap_bin_return_summary.csv'}`",
        f"- High-Low 差值检验：`{result_dir / 'gap_high_low_spread_tests.csv'}`",
        f"- 样本质量表：`{result_dir / 'gap_return_sample_quality.csv'}`",
    ]
    (output_dir / "gap_return_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether A_clean_main leverage Gap bins differ in post-disclosure returns.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rawdata-root", type=Path, default=DEFAULT_RAWDATA_ROOT)
    parser.add_argument("--main-result", type=Path, default=DEFAULT_MAIN_RESULT)
    parser.add_argument("--returns-path", type=Path, default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--min-cluster-n", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.output_dir / "gap_return_test"
    result_dir.mkdir(parents=True, exist_ok=True)

    main_df = read_main_result(args.main_result)
    disclosures = read_disclosure_dates(args.report_path)
    industry = load_industry_map(args.rawdata_root)
    events = main_df.merge(disclosures, on=["firm_id", "period_date"], how="left")
    events["valid_disclosure"] = events["announcement_date"].notna() & (events["announcement_date"] >= events["period_date"])
    events = events[events["valid_disclosure"]].copy()
    events = events.merge(industry, on="firm_id", how="left")
    events["industry_matched"] = events["industry_section_code"].notna()
    events["industry_section_code"] = events["industry_section_code"].fillna("UNKNOWN")
    events["industry_section_name"] = events["industry_section_name"].fillna("UNKNOWN")

    daily = read_daily_returns(args.returns_path)
    events = attach_forward_returns(events, daily, HORIZONS)
    events["signal_year"] = pd.to_datetime(events["signal_trade_date"], errors="coerce").dt.year
    events = assign_gap_bins(events, args.min_cluster_n)

    summary = pd.concat(
        [
            bin_return_summary(events, "gap_tercile", HORIZONS),
            bin_return_summary(events, "gap_quintile", HORIZONS),
        ],
        ignore_index=True,
    )
    spread = pd.concat(
        [
            cluster_spread_test(events, "gap_tercile", "Low Gap", "High Gap", HORIZONS),
            cluster_spread_test(events, "gap_quintile", "Q1 Low Gap", "Q5 High Gap", HORIZONS),
        ],
        ignore_index=True,
    )
    quality = disclosure_quality_table(events)

    keep_cols = [
        "firm_id",
        "period_date",
        "announcement_date",
        "signal_trade_date",
        "period_type",
        "industry_section_code",
        "industry_section_name",
        "industry_code",
        "industry_name",
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "leverage_gap",
        "leverage_status",
        "gap_tercile",
        "gap_quintile",
        "ret_21d",
        "ret_63d",
        "ret_126d",
        "end_date_21d",
        "end_date_63d",
        "end_date_126d",
    ]
    events[[col for col in keep_cols if col in events.columns]].to_csv(result_dir / "gap_return_event_panel.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(result_dir / "gap_bin_return_summary.csv", index=False, encoding="utf-8-sig")
    spread.to_csv(result_dir / "gap_high_low_spread_tests.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(result_dir / "gap_return_sample_quality.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir, result_dir, quality, summary, spread)

    print(f"Events with valid disclosure: {len(events)}")
    print(f"Rows with 21d return: {events['ret_21d'].notna().sum()}")
    print(f"Report: {args.output_dir / 'gap_return_test_report.md'}")
    print(f"Tables: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
