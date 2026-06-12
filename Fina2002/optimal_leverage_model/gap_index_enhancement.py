from __future__ import annotations

import argparse
import io
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from validate_optimal_leverage import load_industry_map, markdown_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "index_enhancement"
DEFAULT_GAP_PATH = Path(__file__).resolve().parent / "output" / "variant_A_clean_main_results.csv"
DEFAULT_ETF_WEIGHT_DIR = PROJECT_ROOT / "etf_weight"
DEFAULT_RAWDATA_ROOT = Path(os.environ.get("QUANT_RAWDATA_ROOT", PROJECT_ROOT / "external_data")).expanduser()
DEFAULT_REPORT_PATH = DEFAULT_RAWDATA_ROOT / "3tables" / "IAR_Rept.csv"
DEFAULT_ALPHAAGENT_DAILY_PATH = Path(
    os.environ.get("QUANT_ALPHAAGENT_DAILY_CSV", PROJECT_ROOT / "external_data" / "alphaagent_qlib_daily_ashare.csv")
).expanduser()

TRADING_DAYS = 252
DEFAULT_TILT_STRENGTHS = [0.25, 0.50, 1.00]
SIGNAL_DIRECTIONS = {
    "low_gap_overweight": 1.0,
    "high_gap_overweight": -1.0,
}
BROAD_INDEX_NAME_MAP = {
    "000300": "CSI 300 / 沪深300",
    "399300": "CSI 300 / 沪深300",
    "000905": "CSI 500 / 中证500",
    "399905": "CSI 500 / 中证500",
    "000906": "CSI 800 / 中证800",
    "399906": "CSI 800 / 中证800",
    "000852": "CSI 1000 / 中证1000",
    "399852": "CSI 1000 / 中证1000",
    "000985": "CSI All Share / 中证全指",
    "399985": "CSI All Share / 中证全指",
    "000010": "SSE 180 / 上证180",
    "399903": "CSI 100 historical venue code / 中证100候选",
    "399904": "CSI 200 historical venue code / 中证200候选",
}


@dataclass(frozen=True)
class EnhancementConfig:
    max_signal_age_days: int
    min_industry_group_n: int
    min_signal_weight_coverage: float
    tilt_strengths: tuple[float, ...]


def normalize_index_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    return digits.zfill(6)


def stock_code_to_firm_id(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(text, errors="coerce")


def parse_index_code_filter(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    return {normalize_index_code(item) for item in str(value).split(",") if item.strip()}


def read_index_weights(
    weight_dir: Path,
    index_code_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    zip_paths = sorted(weight_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No zip files found in {weight_dir}")

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and "IDX_Smprat" in name
            ]
            for csv_name in sorted(csv_names):
                with archive.open(csv_name) as fh:
                    df = pd.read_csv(
                        fh,
                        usecols=["Indexcd", "Enddt", "Stkcd", "Constdnme", "Weight"],
                        dtype={"Indexcd": "string", "Stkcd": "string"},
                        encoding="utf-8-sig",
                        low_memory=False,
                    )
                df["index_code"] = df["Indexcd"].map(normalize_index_code)
                if index_code_filter is not None:
                    df = df[df["index_code"].isin(index_code_filter)].copy()
                    if df.empty:
                        continue
                df["weight_date"] = pd.to_datetime(df["Enddt"], errors="coerce")
                df["firm_id"] = stock_code_to_firm_id(df["Stkcd"])
                df["index_weight"] = pd.to_numeric(df["Weight"], errors="coerce") / 100.0
                df = df.dropna(subset=["index_code", "weight_date", "firm_id", "index_weight"]).copy()
                df = df[df["index_weight"] > 0].copy()
                df["firm_id"] = df["firm_id"].astype("int32")
                df["source_zip"] = zip_path.name
                df["source_file"] = csv_name
                parts.append(
                    df[
                        [
                            "index_code",
                            "weight_date",
                            "firm_id",
                            "Constdnme",
                            "index_weight",
                            "source_zip",
                            "source_file",
                        ]
                    ]
                )
                manifest_rows.append(
                    {
                        "source_zip": zip_path.name,
                        "source_file": csv_name,
                        "rows": int(len(df)),
                        "index_codes": int(df["index_code"].nunique()),
                        "min_date": df["weight_date"].min(),
                        "max_date": df["weight_date"].max(),
                        "stock_ids": int(df["firm_id"].nunique()),
                    }
                )

    if not parts:
        wanted = ", ".join(sorted(index_code_filter)) if index_code_filter else "(all)"
        raise ValueError(f"No index weight rows matched index_code_filter={wanted}")

    weights = pd.concat(parts, ignore_index=True)
    weights = weights.drop_duplicates(["index_code", "weight_date", "firm_id"], keep="last")
    weights = weights.sort_values(["index_code", "weight_date", "firm_id"]).reset_index(drop=True)
    manifest = pd.DataFrame(manifest_rows)
    return weights, manifest


def read_disclosure_dates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Stkcd", "Accper", "Annodt"], low_memory=False)
    df["firm_id"] = pd.to_numeric(df["Stkcd"], errors="coerce")
    df["period_date"] = pd.to_datetime(df["Accper"], errors="coerce")
    df["announcement_date"] = pd.to_datetime(df["Annodt"], errors="coerce")
    df = df.dropna(subset=["firm_id", "period_date", "announcement_date"]).copy()
    df["firm_id"] = df["firm_id"].astype("int32")
    return (
        df.groupby(["firm_id", "period_date"], as_index=False)
        .agg(announcement_date=("announcement_date", "min"))
    )


def read_gap_signals(gap_path: Path, report_path: Path, trading_dates: np.ndarray) -> pd.DataFrame:
    usecols = [
        "firm_id",
        "period_date",
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "leverage_gap",
        "leverage_status",
    ]
    gap = pd.read_csv(gap_path, usecols=usecols)
    gap["firm_id"] = pd.to_numeric(gap["firm_id"], errors="coerce")
    gap["period_date"] = pd.to_datetime(gap["period_date"], errors="coerce")
    gap["leverage_gap"] = pd.to_numeric(gap["leverage_gap"], errors="coerce")
    gap["observed_debt_ratio"] = pd.to_numeric(gap["observed_debt_ratio"], errors="coerce")
    gap["optimal_debt_ratio"] = pd.to_numeric(gap["optimal_debt_ratio"], errors="coerce")
    gap = gap.dropna(subset=["firm_id", "period_date", "leverage_gap"]).copy()
    gap["firm_id"] = gap["firm_id"].astype("int32")

    disclosure = read_disclosure_dates(report_path)
    gap = gap.merge(disclosure, on=["firm_id", "period_date"], how="left")
    gap["valid_disclosure"] = (
        gap["announcement_date"].notna() & (gap["announcement_date"] >= gap["period_date"])
    )

    ann = gap["announcement_date"].to_numpy(dtype="datetime64[ns]")
    search_keys = ann + np.timedelta64(1, "D")
    pos = np.searchsorted(trading_dates, search_keys, side="left")
    available = np.full(len(gap), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = gap["valid_disclosure"].to_numpy() & (pos < len(trading_dates))
    available[ok] = trading_dates[pos[ok]]
    gap["available_trade_date"] = pd.to_datetime(available)

    gap = gap.dropna(subset=["available_trade_date"]).copy()
    gap = gap.sort_values(["firm_id", "available_trade_date", "period_date"])
    gap = gap.drop_duplicates(["firm_id", "available_trade_date"], keep="last")
    return gap[
        [
            "firm_id",
            "period_date",
            "announcement_date",
            "available_trade_date",
            "observed_debt_ratio",
            "optimal_debt_ratio",
            "leverage_gap",
            "leverage_status",
        ]
    ]


def load_stock_next_returns(
    daily_path: Path,
    firm_ids: set[int],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    usecols = ["date", "code", "return"]
    for chunk in pd.read_csv(daily_path, usecols=usecols, chunksize=1_000_000):
        chunk["firm_id"] = stock_code_to_firm_id(chunk["code"])
        chunk["trade_date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["stock_return"] = pd.to_numeric(chunk["return"], errors="coerce")
        chunk = chunk.dropna(subset=["firm_id", "trade_date", "stock_return"])
        chunk["firm_id"] = chunk["firm_id"].astype("int32")
        chunk = chunk[
            chunk["firm_id"].isin(firm_ids)
            & (chunk["trade_date"] >= min_date)
            & (chunk["trade_date"] <= max_date)
            & np.isfinite(chunk["stock_return"])
            & (chunk["stock_return"] > -1.0)
        ].copy()
        if not chunk.empty:
            parts.append(chunk[["firm_id", "trade_date", "stock_return"]])
    if not parts:
        raise ValueError("No stock return rows matched the index constituents.")

    stock = pd.concat(parts, ignore_index=True)
    stock = stock.sort_values(["firm_id", "trade_date"]).drop_duplicates(
        ["firm_id", "trade_date"], keep="last"
    )
    stock["next_return_1d"] = stock.groupby("firm_id", sort=False)["stock_return"].shift(-1)
    stock["next_trade_date"] = stock.groupby("firm_id", sort=False)["trade_date"].shift(-1)
    stock = stock.dropna(subset=["next_return_1d", "next_trade_date"])
    return stock[["firm_id", "trade_date", "next_trade_date", "next_return_1d"]]


def attach_latest_gap(panel: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    left = panel.copy()
    right = signals.copy()
    left["weight_date"] = pd.to_datetime(left["weight_date"], errors="coerce").astype("datetime64[ns]")
    right["available_trade_date"] = pd.to_datetime(
        right["available_trade_date"], errors="coerce"
    ).astype("datetime64[ns]")
    left = left.sort_values(["weight_date", "firm_id"]).reset_index(drop=True)
    right = right.sort_values(["available_trade_date", "firm_id"]).reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        right,
        left_on="weight_date",
        right_on="available_trade_date",
        by="firm_id",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def add_gap_score(panel: pd.DataFrame, min_industry_group_n: int, max_signal_age_days: int) -> pd.DataFrame:
    out = panel.copy()
    out["days_since_gap_available"] = (
        out["weight_date"] - out["available_trade_date"]
    ).dt.days
    out["has_fresh_gap_signal"] = (
        out["leverage_gap"].notna()
        & out["days_since_gap_available"].notna()
        & (out["days_since_gap_available"] >= 0)
        & (out["days_since_gap_available"] <= max_signal_age_days)
    )
    out["industry_section_code"] = out["industry_section_code"].fillna("UNKNOWN").astype(str)
    group_cols = ["index_code", "weight_date", "industry_section_code"]

    valid_gap = out["has_fresh_gap_signal"]
    ranks = out.loc[valid_gap].groupby(group_cols, dropna=False)["leverage_gap"].rank(
        method="first", pct=True
    )
    counts = out.loc[valid_gap].groupby(group_cols, dropna=False)["leverage_gap"].transform("count")
    raw_score = pd.Series(np.nan, index=out.index, dtype="float64")
    raw_score.loc[valid_gap] = 0.5 - ranks
    raw_score.loc[valid_gap] = raw_score.loc[valid_gap].where(counts >= min_industry_group_n)

    weighted_raw = (raw_score * out["base_weight"]).where(raw_score.notna())
    numerator = weighted_raw.groupby([out[col] for col in group_cols], dropna=False).transform("sum")
    denominator = (
        out["base_weight"].where(raw_score.notna())
        .groupby([out[col] for col in group_cols], dropna=False)
        .transform("sum")
    )
    weighted_mean = numerator / denominator.replace(0.0, np.nan)
    out["gap_low_score"] = (raw_score - weighted_mean).fillna(0.0).clip(-0.5, 0.5)
    return out


def summarize_daily_returns(daily: pd.DataFrame, min_signal_weight_coverage: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["index_code", "strategy_direction", "tilt_strength"]
    for (index_code, direction, strength), all_grp in daily.groupby(group_keys, sort=True):
        grp = all_grp[
            (all_grp["signal_weight_coverage"] >= min_signal_weight_coverage)
            & (all_grp["active_share"] > 1e-10)
        ].copy()
        active = pd.to_numeric(grp["active_return"], errors="coerce").dropna()
        base = pd.to_numeric(grp["base_return"], errors="coerce").dropna()
        enhanced = pd.to_numeric(grp["enhanced_return"], errors="coerce").dropna()
        if active.empty:
            rows.append(
                {
                    "index_code": index_code,
                    "strategy_direction": direction,
                    "tilt_strength": strength,
                    "n_eval_days": 0,
                    "n_all_days": int(len(all_grp)),
                    "start_date": pd.NaT,
                    "end_date": pd.NaT,
                    "base_cum_return": np.nan,
                    "enhanced_cum_return": np.nan,
                    "excess_cum_return": np.nan,
                    "ann_active_return": np.nan,
                    "tracking_error": np.nan,
                    "information_ratio": np.nan,
                    "active_win_rate": np.nan,
                    "avg_constituents": float(all_grp["n_constituents"].mean()),
                    "avg_signal_weight_coverage": float(all_grp["signal_weight_coverage"].mean()),
                    "avg_active_share": float(all_grp["active_share"].mean()),
                    "avg_low_gap_score_weighted": float(all_grp["weighted_gap_low_score"].mean()),
                    "coverage_filter": min_signal_weight_coverage,
                }
            )
            continue
        tracking_error = float(active.std(ddof=1) * math.sqrt(TRADING_DAYS))
        mean_active = float(active.mean())
        ann_active = mean_active * TRADING_DAYS
        rows.append(
            {
                "index_code": index_code,
                "strategy_direction": direction,
                "tilt_strength": strength,
                "n_eval_days": int(len(active)),
                "n_all_days": int(len(all_grp)),
                "start_date": grp["weight_date"].min(),
                "end_date": grp["weight_date"].max(),
                "base_cum_return": float(np.prod(1.0 + base) - 1.0),
                "enhanced_cum_return": float(np.prod(1.0 + enhanced) - 1.0),
                "excess_cum_return": float(np.prod(1.0 + active) - 1.0),
                "ann_active_return": ann_active,
                "tracking_error": tracking_error,
                "information_ratio": ann_active / tracking_error if tracking_error > 0 else np.nan,
                "active_win_rate": float((active > 0).mean()),
                "avg_constituents": float(grp["n_constituents"].mean()),
                "avg_signal_weight_coverage": float(grp["signal_weight_coverage"].mean()),
                "avg_active_share": float(grp["active_share"].mean()),
                "avg_low_gap_score_weighted": float(grp["weighted_gap_low_score"].mean()),
                "coverage_filter": min_signal_weight_coverage,
            }
        )
    return pd.DataFrame(rows)


def build_enhancement_returns(panel: pd.DataFrame, config: EnhancementConfig) -> pd.DataFrame:
    base_group_cols = ["index_code", "weight_date"]
    panel["weighted_base_return"] = panel["base_weight"] * panel["next_return_1d"]
    base_daily = (
        panel.groupby(base_group_cols, dropna=False)
        .agg(
            base_return=("weighted_base_return", "sum"),
            n_constituents=("firm_id", "size"),
            n_fresh_gap=("has_fresh_gap_signal", "sum"),
            signal_weight_coverage=(
                "base_weight",
                lambda s: float(s[panel.loc[s.index, "has_fresh_gap_signal"]].sum()),
            ),
            weighted_gap_low_score=(
                "gap_low_score",
                lambda s: float(np.sum(s * panel.loc[s.index, "base_weight"])),
            ),
        )
        .reset_index()
    )

    daily_parts: list[pd.DataFrame] = []
    for direction_name, direction_sign in SIGNAL_DIRECTIONS.items():
        for strength in config.tilt_strengths:
            tmp = panel[["index_code", "weight_date", "base_weight", "next_return_1d", "gap_low_score"]].copy()
            tmp["tilt_multiplier"] = 1.0 + float(strength) * direction_sign * tmp["gap_low_score"]
            tmp["enhanced_weight_raw"] = tmp["base_weight"] * tmp["tilt_multiplier"]
            denom = tmp.groupby(base_group_cols, dropna=False)["enhanced_weight_raw"].transform("sum")
            tmp["enhanced_weight"] = tmp["enhanced_weight_raw"] / denom.replace(0.0, np.nan)
            tmp["weighted_enhanced_return"] = tmp["enhanced_weight"] * tmp["next_return_1d"]
            tmp["abs_active_weight"] = (tmp["enhanced_weight"] - tmp["base_weight"]).abs()
            enh_daily = (
                tmp.groupby(base_group_cols, dropna=False)
                .agg(
                    enhanced_return=("weighted_enhanced_return", "sum"),
                    active_share=("abs_active_weight", lambda s: float(0.5 * s.sum())),
                )
                .reset_index()
            )
            daily = base_daily.merge(enh_daily, on=base_group_cols, how="left")
            daily["strategy_direction"] = direction_name
            daily["tilt_strength"] = float(strength)
            daily["active_return"] = daily["enhanced_return"] - daily["base_return"]
            daily_parts.append(daily)
    return pd.concat(daily_parts, ignore_index=True)


def write_report(
    output_dir: Path,
    source_manifest: pd.DataFrame,
    index_summary: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    config: EnhancementConfig,
) -> None:
    best = summary[summary["n_eval_days"] > 0].sort_values(
        ["information_ratio", "ann_active_return"],
        ascending=[False, False],
    ).head(20)
    by_strength = (
        summary.groupby(["strategy_direction", "tilt_strength"], as_index=False)
        .agg(
            n_indices=("index_code", "nunique"),
            n_indices_with_eval=("n_eval_days", lambda s: int((s > 0).sum())),
            mean_ann_active_return=("ann_active_return", "mean"),
            mean_tracking_error=("tracking_error", "mean"),
            mean_information_ratio=("information_ratio", "mean"),
            mean_win_rate=("active_win_rate", "mean"),
            mean_signal_weight_coverage=("avg_signal_weight_coverage", "mean"),
            mean_active_share=("avg_active_share", "mean"),
        )
        .sort_values("tilt_strength")
    )
    overall_daily = (
        daily[
            (daily["signal_weight_coverage"] >= config.min_signal_weight_coverage)
            & (daily["active_share"] > 1e-10)
        ]
        .groupby(["weight_date", "strategy_direction", "tilt_strength"], as_index=False)["active_return"]
        .mean()
        .rename(columns={"active_return": "mean_index_active_return"})
    )
    overall_rows = []
    for (direction, strength), grp in overall_daily.groupby(["strategy_direction", "tilt_strength"]):
        active = grp["mean_index_active_return"].dropna()
        te = float(active.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(active) > 1 else np.nan
        ann = float(active.mean() * TRADING_DAYS) if len(active) else np.nan
        overall_rows.append(
            {
                "strategy_direction": direction,
                "tilt_strength": strength,
                "n_days": int(len(active)),
                "mean_index_ann_active_return": ann,
                "mean_index_tracking_error": te,
                "mean_index_information_ratio": ann / te if te and np.isfinite(te) else np.nan,
                "mean_index_win_rate": float((active > 0).mean()) if len(active) else np.nan,
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(output_dir / "index_enhancement_overall_by_strength.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Gap 指数增强检验报告",
        "",
        "## 方法",
        "",
        "- 信号使用冻结主口径 `variant_A_clean_main_results.csv` 中的 `leverage_gap`。",
        "- 信息可用日使用 `IAR_Rept.csv` 的 `Annodt`，并从公告日后的第一个交易日开始允许使用 Gap。",
        "- 指数权重使用 `etf_weight` 下 RESSET `IDX_Smprat` 文件，权重日期 `Enddt` 当作组合形成日。",
        "- 为避免看未来，组合在 `Enddt` 使用当日已可得的最新 Gap，并评价下一交易日个股收益。",
        "- 增强方向为低 Gap 倾斜：在指数成分股内部、同行业分组内上调较低 Gap 股票权重，下调较高 Gap 股票权重。",
        "- 同时输出反向版本 `high_gap_overweight`，用于检查信号方向是否稳定。",
        "- 权重调整是 long-only 相对基准倾斜，不做卖空；行业内分数做基准权重加权去均值，尽量保留行业暴露。",
        "- 当前结果为毛收益，没有扣除交易成本、冲击成本和复制误差。",
        "",
        "## 参数",
        "",
        f"- `max_signal_age_days`: {config.max_signal_age_days}",
        f"- `min_industry_group_n`: {config.min_industry_group_n}",
        f"- `min_signal_weight_coverage`: {config.min_signal_weight_coverage}",
        f"- `tilt_strengths`: {list(config.tilt_strengths)}",
        "",
        "## 指数权重来源",
        "",
        markdown_table(source_manifest, max_rows=20),
        "",
        "## 指数覆盖",
        "",
        markdown_table(index_summary, max_rows=30),
        "",
        "## 不同倾斜强度的平均表现",
        "",
        f"下表只统计 `signal_weight_coverage >= {config.min_signal_weight_coverage:.0%}` 且主动权重非零的日期。",
        "",
        markdown_table(by_strength, max_rows=20),
        "",
        "## 跨指数平均日度主动收益",
        "",
        f"下表同样只使用 `signal_weight_coverage >= {config.min_signal_weight_coverage:.0%}` 的指数日期。",
        "",
        markdown_table(overall, max_rows=20),
        "",
        "## 表现最好的指数-强度组合",
        "",
        markdown_table(best, max_rows=20),
        "",
        "## 输出文件",
        "",
        "- `index_weight_manifest.csv`: 权重压缩包与 CSV 来源清单。",
        "- `index_weight_index_summary.csv`: 指数代码、日期范围、成分数量和覆盖率。",
        "- `gap_index_enhancement_daily_returns.csv`: 每个指数、日期、倾斜强度的基准收益、增强收益和主动收益。",
        "- `gap_index_enhancement_summary.csv`: 在最低信号覆盖率过滤后的指数层面年化主动收益、跟踪误差、信息比率和覆盖率。",
        "- `index_enhancement_overall_by_strength.csv`: 对所有指数等权平均后的主动收益摘要。",
        "",
        "## 解释口径",
        "",
        "`low_gap_overweight` 中的低 Gap 包括 `Gap<0` 的 under-levered 公司，也包括同行业同指数内相对 Gap 更低的公司。"
        "`high_gap_overweight` 是完全反向的方向检验。增强逻辑不是认为 `d_obs` 越低越好，而是检验企业相对自身结构性目标负债率的偏离是否能改善指数成分股内排序。",
        "",
        "正式用于指数增强前，还需要加入交易成本、换手约束、调仓频率约束和样本外参数选择。当前报告用于判断 Gap 能否作为指数成分股内的增强信号。",
    ]
    (output_dir / "gap_index_enhancement_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EnhancementConfig(
        max_signal_age_days=int(args.max_signal_age_days),
        min_industry_group_n=int(args.min_industry_group_n),
        min_signal_weight_coverage=float(args.min_signal_weight_coverage),
        tilt_strengths=tuple(float(x) for x in args.tilt_strengths.split(",")),
    )

    index_code_filter = parse_index_code_filter(args.index_codes)
    weights, source_manifest = read_index_weights(Path(args.etf_weight_dir), index_code_filter)
    source_manifest.to_csv(output_dir / "index_weight_manifest.csv", index=False, encoding="utf-8-sig")

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
    panel = add_gap_score(panel, config.min_industry_group_n, config.max_signal_age_days)

    daily = build_enhancement_returns(panel, config)
    summary = summarize_daily_returns(daily, config.min_signal_weight_coverage)

    index_summary = (
        daily.groupby("index_code", as_index=False)
        .agg(
            n_days=("weight_date", "nunique"),
            start_date=("weight_date", "min"),
            end_date=("weight_date", "max"),
            avg_constituents=("n_constituents", "mean"),
            avg_signal_weight_coverage=("signal_weight_coverage", "mean"),
            avg_base_return=("base_return", "mean"),
        )
        .sort_values(["n_days", "index_code"], ascending=[False, True])
    )

    daily.to_csv(output_dir / "gap_index_enhancement_daily_returns.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "gap_index_enhancement_summary.csv", index=False, encoding="utf-8-sig")
    index_summary.to_csv(output_dir / "index_weight_index_summary.csv", index=False, encoding="utf-8-sig")

    write_report(output_dir, source_manifest, index_summary, daily, summary, config)

    best = (
        summary[summary["n_eval_days"] > 0]
        .sort_values(["information_ratio", "ann_active_return"], ascending=[False, False])
        .head(5)
    )
    print("Gap index enhancement completed.")
    print(f"Index weight files: {len(source_manifest)} CSV files from {args.etf_weight_dir}")
    print(f"Index code filter: {sorted(index_code_filter) if index_code_filter else 'all'}")
    print(f"Index codes: {daily['index_code'].nunique()}")
    print(f"Daily index-strength rows: {len(daily)}")
    print("Best combinations:")
    print(
        best[
            [
                "index_code",
                "strategy_direction",
                "tilt_strength",
                "n_eval_days",
                "ann_active_return",
                "tracking_error",
                "information_ratio",
                "active_win_rate",
            ]
        ].to_string(index=False)
    )
    print(f"Summary: {output_dir / 'gap_index_enhancement_summary.csv'}")
    print(f"Report: {output_dir / 'gap_index_enhancement_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use leverage Gap as an index enhancement component.")
    parser.add_argument("--gap-path", default=str(DEFAULT_GAP_PATH))
    parser.add_argument("--etf-weight-dir", default=str(DEFAULT_ETF_WEIGHT_DIR))
    parser.add_argument("--rawdata-root", default=str(DEFAULT_RAWDATA_ROOT))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--daily-returns-path", default=str(DEFAULT_ALPHAAGENT_DAILY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-signal-age-days", type=int, default=540)
    parser.add_argument("--min-industry-group-n", type=int, default=5)
    parser.add_argument("--min-signal-weight-coverage", type=float, default=0.50)
    parser.add_argument("--tilt-strengths", default="0.25,0.50,1.00")
    parser.add_argument(
        "--index-codes",
        default="",
        help="Comma-separated normalized index codes to keep, e.g. 000906,000010.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
