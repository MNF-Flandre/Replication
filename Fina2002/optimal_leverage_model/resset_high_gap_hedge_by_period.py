from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from validate_optimal_leverage import load_industry_map, markdown_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_GAP_PATH = MODEL_DIR / "output" / "variant_A_clean_main_results.csv"
DEFAULT_REPT_PATH = Path(
    os.environ.get("QUANT_REPT_PATH", PROJECT_ROOT / "external_data" / "3tables" / "IAR_Rept.csv")
).expanduser()
DEFAULT_RAWDATA_ROOT = Path(os.environ.get("QUANT_RAWDATA_ROOT", PROJECT_ROOT / "external_data")).expanduser()
DEFAULT_INDEX_RETURN_PATH = (
    PROJECT_ROOT
    / "etf_weight"
    / "output"
    / "resset_index_returns"
    / "resset_index_daily_returns.csv"
)
DEFAULT_COMPONENT_PATH = (
    PROJECT_ROOT
    / "etf_weight"
    / "output"
    / "resset_index_returns"
    / "resset_component_intervals.csv"
)
DEFAULT_MARKET_DAILY_ROOT = Path(
    os.environ.get("QUANT_MARKET_DAILY_ROOT", PROJECT_ROOT / "external_data" / "market_daily")
).expanduser()
DEFAULT_OUTPUT_DIR = MODEL_DIR / "output" / "resset_high_gap_hedge_by_period"
TRADING_DAYS = 252


@dataclass(frozen=True)
class HedgeByPeriodConfig:
    high_quantile: float
    hedge_notional: float
    max_signal_age_days: int
    min_signal_weight_coverage: float
    min_fresh_names: int
    min_industry_group_n: int


def normalize_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text


def parse_index_codes(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    return {normalize_code(item) for item in str(value).split(",") if item.strip()}


def list_market_parquets(root: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[str]:
    paths = sorted(root.glob("year=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root}")
    start_year = int(start_date.year)
    end_year = int(end_date.year)
    selected = []
    for path in paths:
        parent = path.parent.name
        if not parent.startswith("year="):
            continue
        try:
            year = int(parent.split("=", 1)[1])
        except ValueError:
            continue
        if start_year <= year <= end_year:
            selected.append(str(path))
    if not selected:
        raise FileNotFoundError(f"No parquet files matched {start_year}-{end_year} under {root}")
    return selected


def read_resset_benchmark(path: Path, index_codes: set[str] | None) -> pd.DataFrame:
    usecols = [
        "index_code",
        "index_name",
        "trade_date",
        "market_cap_weighted_return",
        "equal_weight_return",
    ]
    df = pd.read_csv(path, usecols=usecols, dtype={"index_code": "string"})
    df["index_code"] = df["index_code"].map(normalize_code)
    if index_codes is not None:
        df = df[df["index_code"].isin(index_codes)].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["base_return_resset"] = pd.to_numeric(df["market_cap_weighted_return"], errors="coerce")
    df["equal_weight_return_resset"] = pd.to_numeric(df["equal_weight_return"], errors="coerce")
    df = df.dropna(subset=["index_code", "trade_date", "base_return_resset"]).copy()
    return df[
        [
            "index_code",
            "index_name",
            "trade_date",
            "base_return_resset",
            "equal_weight_return_resset",
        ]
    ]


def read_components(path: Path, index_codes: set[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"index_code": "string", "stock_id": "string"},
        parse_dates=["begin_date", "end_date_filled"],
    )
    df["index_code"] = df["index_code"].map(normalize_code)
    df["stock_id"] = df["stock_id"].map(normalize_code)
    df = df[df["index_code"].isin(index_codes)].copy()
    df = df.dropna(subset=["index_code", "stock_id", "begin_date", "end_date_filled"])
    if df.empty:
        wanted = ", ".join(sorted(index_codes))
        raise ValueError(f"No RESSET component rows matched index codes: {wanted}")
    return df


def read_disclosure_dates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Stkcd", "Accper", "Annodt"], low_memory=False)
    df["stock_id"] = df["Stkcd"].map(normalize_code)
    df["period_date"] = pd.to_datetime(df["Accper"], errors="coerce")
    df["announcement_date"] = pd.to_datetime(df["Annodt"], errors="coerce")
    df = df.dropna(subset=["stock_id", "period_date", "announcement_date"]).copy()
    return (
        df.groupby(["stock_id", "period_date"], as_index=False)
        .agg(announcement_date=("announcement_date", "min"))
    )


def read_gap_signals(
    gap_path: Path,
    report_path: Path,
    trading_dates: np.ndarray,
) -> pd.DataFrame:
    usecols = [
        "firm_id",
        "period_date",
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "leverage_gap",
        "leverage_status",
    ]
    gap = pd.read_csv(gap_path, usecols=usecols)
    gap["stock_id"] = gap["firm_id"].map(normalize_code)
    gap["period_date"] = pd.to_datetime(gap["period_date"], errors="coerce")
    for col in ["observed_debt_ratio", "optimal_debt_ratio", "leverage_gap"]:
        gap[col] = pd.to_numeric(gap[col], errors="coerce")
    gap = gap.dropna(subset=["stock_id", "period_date", "leverage_gap"]).copy()

    disclosure = read_disclosure_dates(report_path)
    gap = gap.merge(disclosure, on=["stock_id", "period_date"], how="left")
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

    gap = gap.sort_values(["stock_id", "available_trade_date", "period_date"])
    gap = gap.drop_duplicates(["stock_id", "available_trade_date"], keep="last")
    return gap[
        [
            "stock_id",
            "period_date",
            "announcement_date",
            "available_trade_date",
            "observed_debt_ratio",
            "optimal_debt_ratio",
            "leverage_gap",
            "leverage_status",
        ]
    ]


def read_industry(rawdata_root: Path, stock_ids: list[str]) -> pd.DataFrame:
    industry = load_industry_map(rawdata_root)
    if industry.empty:
        return pd.DataFrame(
            {
                "stock_id": stock_ids,
                "industry_section_code": ["UNKNOWN"] * len(stock_ids),
                "industry_section_name": ["UNKNOWN"] * len(stock_ids),
            }
        )
    industry = industry.copy()
    industry["stock_id"] = industry["firm_id"].map(normalize_code)
    industry["industry_section_code"] = (
        industry["industry_section_code"].fillna("UNKNOWN").astype(str)
    )
    industry["industry_section_name"] = (
        industry["industry_section_name"].fillna("UNKNOWN").astype(str)
    )
    return industry[["stock_id", "industry_section_code", "industry_section_name"]]


def build_component_market_panel(
    components: pd.DataFrame,
    parquet_paths: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pl.DataFrame:
    stock_ids = sorted(components["stock_id"].dropna().unique().tolist())
    comp_pl = pl.DataFrame(
        {
            "index_code": components["index_code"].astype(str).tolist(),
            "index_name": components["index_name"].fillna("").astype(str).tolist(),
            "stock_id": components["stock_id"].astype(str).tolist(),
            "begin_date": components["begin_date"].dt.strftime("%Y-%m-%d").tolist(),
            "end_date_filled": components["end_date_filled"].dt.strftime("%Y-%m-%d").tolist(),
        }
    ).with_columns(
        [
            pl.col("begin_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col("end_date_filled").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        ]
    )

    market = (
        pl.scan_parquet(parquet_paths)
        .select(
            [
                pl.col("stock_id").cast(pl.Utf8),
                pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                pl.col("change_ratio").cast(pl.Float64),
                pl.col("daily_market_cap").cast(pl.Float64),
                pl.col("trade_status").cast(pl.Int64),
            ]
        )
        .filter(pl.col("stock_id").is_in(stock_ids))
        .filter(pl.col("trade_date") >= pl.lit(start_date.date()))
        .filter(pl.col("trade_date") <= pl.lit(end_date.date()))
        .sort(["stock_id", "trade_date"])
        .with_columns(
            pl.col("daily_market_cap")
            .shift(1)
            .over("stock_id")
            .alias("lag_daily_market_cap")
        )
    )

    joined = (
        market.join(comp_pl.lazy(), on="stock_id", how="inner")
        .filter(pl.col("trade_date") >= pl.col("begin_date"))
        .filter(pl.col("trade_date") <= pl.col("end_date_filled"))
        .sort(["index_code", "trade_date", "stock_id", "begin_date"])
        .unique(["index_code", "trade_date", "stock_id"], keep="last")
        .with_columns(
            (
                (pl.col("lag_daily_market_cap") > 0)
                & pl.col("change_ratio").is_not_null()
                & (pl.col("change_ratio") > -1.0)
            ).alias("valid_base_weight")
        )
        .select(
            [
                "index_code",
                "index_name",
                "stock_id",
                "trade_date",
                "change_ratio",
                "lag_daily_market_cap",
                "valid_base_weight",
            ]
        )
    )
    return joined.collect(streaming=True)


def attach_signals_and_groups(
    panel: pl.DataFrame,
    signals: pd.DataFrame,
    industry: pd.DataFrame,
    config: HedgeByPeriodConfig,
) -> pl.DataFrame:
    signal_pl = pl.from_pandas(signals).with_columns(
        [
            pl.col("stock_id").cast(pl.Utf8),
            pl.col("period_date").cast(pl.Date),
            pl.col("announcement_date").cast(pl.Date),
            pl.col("available_trade_date").cast(pl.Date),
            pl.col("leverage_gap").cast(pl.Float64),
            pl.col("observed_debt_ratio").cast(pl.Float64),
            pl.col("optimal_debt_ratio").cast(pl.Float64),
        ]
    )
    industry_pl = pl.from_pandas(industry).with_columns(pl.col("stock_id").cast(pl.Utf8))

    left = panel.sort(["stock_id", "trade_date"])
    right = signal_pl.sort(["stock_id", "available_trade_date"])
    try:
        out = left.join_asof(
            right,
            left_on="trade_date",
            right_on="available_trade_date",
            by="stock_id",
            strategy="backward",
            check_sortedness=False,
        )
    except TypeError:
        out = left.join_asof(
            right,
            left_on="trade_date",
            right_on="available_trade_date",
            by="stock_id",
            strategy="backward",
        )

    out = out.join(industry_pl, on="stock_id", how="left").with_columns(
        [
            pl.col("industry_section_code").fill_null("UNKNOWN").cast(pl.Utf8),
            pl.col("industry_section_name").fill_null("UNKNOWN").cast(pl.Utf8),
            (pl.col("trade_date") - pl.col("available_trade_date"))
            .dt.total_days()
            .alias("days_since_gap_available"),
        ]
    )
    out = out.with_columns(
        (
            pl.col("valid_base_weight")
            & pl.col("leverage_gap").is_not_null()
            & pl.col("days_since_gap_available").is_not_null()
            & (pl.col("days_since_gap_available") >= 0)
            & (pl.col("days_since_gap_available") <= config.max_signal_age_days)
        ).alias("has_fresh_gap_signal")
    )

    group_index = ["index_code", "trade_date"]
    group_industry = ["index_code", "trade_date", "industry_section_code"]
    out = out.with_columns(
        [
            pl.when(pl.col("valid_base_weight"))
            .then(pl.col("lag_daily_market_cap"))
            .otherwise(None)
            .alias("base_weight_raw"),
            pl.when(pl.col("has_fresh_gap_signal"))
            .then(pl.col("leverage_gap").rank("ordinal").over(group_index))
            .otherwise(None)
            .alias("rank_raw_index"),
            pl.when(pl.col("has_fresh_gap_signal"))
            .then(pl.col("leverage_gap").count().over(group_index))
            .otherwise(None)
            .alias("count_raw_index"),
            pl.when(pl.col("has_fresh_gap_signal"))
            .then(pl.col("leverage_gap").rank("ordinal").over(group_industry))
            .otherwise(None)
            .alias("rank_industry_neutral"),
            pl.when(pl.col("has_fresh_gap_signal"))
            .then(pl.col("leverage_gap").count().over(group_industry))
            .otherwise(None)
            .alias("count_industry_neutral"),
        ]
    )
    out = out.with_columns(
        [
            (
                pl.col("base_weight_raw")
                / pl.col("base_weight_raw").sum().over(group_index)
            ).alias("base_weight"),
            (
                pl.col("has_fresh_gap_signal")
                & (pl.col("count_raw_index") >= config.min_fresh_names)
                & (
                    (pl.col("rank_raw_index") / pl.col("count_raw_index"))
                    >= (1.0 - config.high_quantile)
                )
            ).alias("high_gap_raw_index"),
            (
                pl.col("has_fresh_gap_signal")
                & (pl.col("count_industry_neutral") >= config.min_industry_group_n)
                & (
                    (
                        pl.col("rank_industry_neutral")
                        / pl.col("count_industry_neutral")
                    )
                    >= (1.0 - config.high_quantile)
                )
            ).alias("high_gap_industry_neutral"),
        ]
    )
    return out


def compute_daily_for_method(panel: pl.DataFrame, method: str) -> pl.DataFrame:
    high_col = f"high_gap_{method}"
    weighted_return = pl.col("base_weight") * pl.col("change_ratio")
    high_weighted_return = pl.when(pl.col(high_col)).then(weighted_return).otherwise(0.0)
    high_weight = pl.when(pl.col(high_col)).then(pl.col("base_weight")).otherwise(0.0)
    rest_weighted_return = pl.when(~pl.col(high_col)).then(weighted_return).otherwise(0.0)
    rest_weight = pl.when(~pl.col(high_col)).then(pl.col("base_weight")).otherwise(0.0)

    daily = (
        panel.group_by(["index_code", "index_name", "trade_date"])
        .agg(
            [
                weighted_return.sum().alias("base_return_recomputed"),
                pl.col("stock_id").n_unique().alias("n_constituents"),
                pl.col("has_fresh_gap_signal").sum().alias("fresh_names"),
                pl.when(pl.col("has_fresh_gap_signal"))
                .then(pl.col("base_weight"))
                .otherwise(0.0)
                .sum()
                .alias("signal_weight_coverage"),
                pl.col(high_col).sum().alias("high_names"),
                high_weight.sum().alias("high_weight"),
                high_weighted_return.sum().alias("high_weighted_return"),
                rest_weight.sum().alias("rest_weight"),
                rest_weighted_return.sum().alias("rest_weighted_return"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("high_weight") > 0)
                .then(pl.col("high_weighted_return") / pl.col("high_weight"))
                .otherwise(None)
                .alias("high_gap_return"),
                pl.when(pl.col("rest_weight") > 0)
                .then(pl.col("rest_weighted_return") / pl.col("rest_weight"))
                .otherwise(None)
                .alias("no_high_return"),
                pl.lit(method).alias("selection_method"),
            ]
        )
    )
    return daily


def build_daily_hedge_returns(
    components: pd.DataFrame,
    benchmark: pd.DataFrame,
    signals: pd.DataFrame,
    industry: pd.DataFrame,
    parquet_paths: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: HedgeByPeriodConfig,
) -> pd.DataFrame:
    panel = build_component_market_panel(components, parquet_paths, start_date, end_date)
    panel = attach_signals_and_groups(panel, signals, industry, config)

    daily = pl.concat(
        [
            compute_daily_for_method(panel, "raw_index"),
            compute_daily_for_method(panel, "industry_neutral"),
        ],
        how="vertical",
    ).collect() if isinstance(panel, pl.LazyFrame) else pl.concat(
        [
            compute_daily_for_method(panel, "raw_index"),
            compute_daily_for_method(panel, "industry_neutral"),
        ],
        how="vertical",
    )
    daily_pd = daily.to_pandas()
    daily_pd["trade_date"] = pd.to_datetime(daily_pd["trade_date"])
    daily_pd = daily_pd.merge(
        benchmark,
        on=["index_code", "index_name", "trade_date"],
        how="left",
    )
    daily_pd = daily_pd.dropna(subset=["base_return_resset"]).copy()
    daily_pd = daily_pd[
        (daily_pd["signal_weight_coverage"] >= config.min_signal_weight_coverage)
        & (daily_pd["fresh_names"] >= config.min_fresh_names)
        & (daily_pd["high_names"] > 0)
        & (daily_pd["high_weight"] > 0)
    ].copy()
    daily_pd["remove_high_active_vs_recomputed"] = (
        daily_pd["no_high_return"] - daily_pd["base_return_recomputed"]
    )
    daily_pd["remove_high_active_vs_resset"] = (
        daily_pd["no_high_return"] - daily_pd["base_return_resset"]
    )
    daily_pd["high_gap_excess_vs_resset"] = (
        daily_pd["high_gap_return"] - daily_pd["base_return_resset"]
    )
    daily_pd["short_high_overlay_active"] = (
        -config.hedge_notional * daily_pd["high_gap_excess_vs_resset"]
    )
    daily_pd["year"] = daily_pd["trade_date"].dt.year
    daily_pd["quarter"] = daily_pd["trade_date"].dt.to_period("Q").astype(str)
    daily_pd["calendar_block"] = pd.cut(
        daily_pd["year"],
        bins=[-np.inf, 2019, 2021, 2022, 2024, np.inf],
        labels=["through_2019", "2020_2021", "2022", "2023_2024", "after_2024"],
    ).astype(str)
    daily_pd["daily_market_state"] = np.where(
        daily_pd["base_return_resset"] >= 0,
        "up_day",
        "down_day",
    )
    return daily_pd.sort_values(["index_code", "selection_method", "trade_date"]).reset_index(drop=True)


def summarize_return_series(series: pd.Series) -> dict[str, float | int]:
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


def summarize_by(
    daily: pd.DataFrame,
    group_cols: list[str],
    return_cols: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in daily.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        common = dict(zip(group_cols, keys))
        common.update(
            {
                "avg_signal_weight_coverage": float(group["signal_weight_coverage"].mean()),
                "avg_fresh_names": float(group["fresh_names"].mean()),
                "avg_high_names": float(group["high_names"].mean()),
                "avg_high_weight": float(group["high_weight"].mean()),
                "base_cumulative_return": float(
                    np.prod(1.0 + pd.to_numeric(group["base_return_resset"], errors="coerce").dropna())
                    - 1.0
                ),
                "base_mean_daily": float(group["base_return_resset"].mean()),
            }
        )
        for col, label in return_cols.items():
            rows.append({"strategy_metric": label, **common, **summarize_return_series(group[col])})
    return pd.DataFrame(rows)


def summarize_cross_index_by_date(
    daily: pd.DataFrame,
    group_cols: list[str],
    return_cols: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in daily.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_date = (
            group.groupby("trade_date", as_index=False)
            .agg(
                {
                    **{col: "mean" for col in return_cols},
                    "base_return_resset": "mean",
                    "signal_weight_coverage": "mean",
                    "fresh_names": "mean",
                    "high_names": "mean",
                    "high_weight": "mean",
                    "index_code": "nunique",
                }
            )
            .rename(columns={"index_code": "n_indices_on_date"})
        )
        common = dict(zip(group_cols, keys))
        common.update(
            {
                "n_index_obs": int(len(group)),
                "n_indices": int(group["index_code"].nunique()),
                "avg_indices_per_day": float(by_date["n_indices_on_date"].mean()),
                "avg_signal_weight_coverage": float(by_date["signal_weight_coverage"].mean()),
                "avg_fresh_names": float(by_date["fresh_names"].mean()),
                "avg_high_names": float(by_date["high_names"].mean()),
                "avg_high_weight": float(by_date["high_weight"].mean()),
                "base_cumulative_return": float(np.prod(1.0 + by_date["base_return_resset"]) - 1.0),
                "base_mean_daily": float(by_date["base_return_resset"].mean()),
            }
        )
        for col, label in return_cols.items():
            rows.append({"strategy_metric": label, **common, **summarize_return_series(by_date[col])})
    return pd.DataFrame(rows)


def add_period_market_state(
    daily: pd.DataFrame,
    period_col: str,
) -> pd.DataFrame:
    state = (
        daily.groupby(["index_code", period_col], as_index=False)["base_return_resset"]
        .agg(lambda s: float(np.prod(1.0 + s.dropna()) - 1.0))
        .rename(columns={"base_return_resset": f"{period_col}_base_cumulative_return"})
    )
    state[f"{period_col}_market_state"] = np.where(
        state[f"{period_col}_base_cumulative_return"] >= 0,
        f"up_{period_col}",
        f"down_{period_col}",
    )
    return daily.merge(state, on=["index_code", period_col], how="left")


def write_report(
    output_dir: Path,
    daily: pd.DataFrame,
    overall: pd.DataFrame,
    by_year: pd.DataFrame,
    by_year_state: pd.DataFrame,
    by_block: pd.DataFrame,
    by_quarter: pd.DataFrame,
    config: HedgeByPeriodConfig,
) -> None:
    main_metric = "remove_high_active_vs_resset"
    overall_main = overall[overall["strategy_metric"].eq(main_metric)].sort_values(
        ["selection_method", "information_ratio"],
        ascending=[True, False],
    )
    year_main = by_year[by_year["strategy_metric"].eq(main_metric)].copy()
    year_cross = (
        year_main.groupby(["selection_method", "year"], as_index=False)
        .agg(
            n_indices=("index_code", "nunique"),
            n_days=("n_days", "sum"),
            mean_ann_active_return=("ann_return_arithmetic", "mean"),
            mean_information_ratio=("information_ratio", "mean"),
            mean_cumulative_return=("cumulative_return", "mean"),
            mean_high_weight=("avg_high_weight", "mean"),
        )
        .sort_values(["selection_method", "year"])
    )
    quarter_main = by_quarter[by_quarter["strategy_metric"].eq(main_metric)].copy()
    quarter_best_worst = pd.concat(
        [
            quarter_main.sort_values("cumulative_return", ascending=False).head(10),
            quarter_main.sort_values("cumulative_return", ascending=True).head(10),
        ],
        ignore_index=True,
    )
    state_main = by_year_state[by_year_state["strategy_metric"].eq(main_metric)].sort_values(
        ["selection_method", "year_market_state"]
    )
    block_main = by_block[by_block["strategy_metric"].eq(main_metric)].sort_values(
        ["selection_method", "calendar_block"]
    )

    lines = [
        "# RESSET high Gap hedge by period",
        "",
        "## Test design",
        "",
        "- Gap signal: frozen A_clean_main result, `variant_A_clean_main_results.csv`.",
        "- Benchmark: RESSET reconstructed broad index daily `market_cap_weighted_return`.",
        "- Information timing: a report-period Gap is tradable only from the first trading day after `Annodt`.",
        "- Daily signal: each stock uses the latest available disclosed Gap as of the index date.",
        "- `raw_index`: top Gap stocks selected inside the index constituent pool.",
        "- `industry_neutral`: top Gap stocks selected inside index-date-industry cells.",
        "- Main long-only metric: `remove_high_active_vs_resset = no_high_return - RESSET_market_cap_weighted_return`.",
        "- Overlay metric: `short_high_overlay_active = -hedge_notional * (high_gap_return - RESSET_market_cap_weighted_return)`.",
        "",
        "## Parameters",
        "",
        f"- `high_quantile`: {config.high_quantile}",
        f"- `hedge_notional`: {config.hedge_notional}",
        f"- `max_signal_age_days`: {config.max_signal_age_days}",
        f"- `min_signal_weight_coverage`: {config.min_signal_weight_coverage}",
        f"- `min_fresh_names`: {config.min_fresh_names}",
        f"- `min_industry_group_n`: {config.min_industry_group_n}",
        "",
        "## Overall main result",
        "",
        markdown_table(overall_main, max_rows=30),
        "",
        "## Cross-index average by year",
        "",
        markdown_table(year_cross, max_rows=80),
        "",
        "## Year market-state split",
        "",
        markdown_table(state_main, max_rows=40),
        "",
        "## Calendar block split",
        "",
        markdown_table(block_main, max_rows=40),
        "",
        "## Best and worst index-quarters",
        "",
        markdown_table(quarter_best_worst, max_rows=30),
        "",
        "## Output files",
        "",
        "- `resset_high_gap_hedge_daily_returns.csv`: index-date-method daily hedge returns.",
        "- `resset_high_gap_hedge_overall.csv`: index-level and method-level summary.",
        "- `resset_high_gap_hedge_by_year.csv`: index-year summary.",
        "- `resset_high_gap_hedge_by_quarter.csv`: index-quarter summary.",
        "- `resset_high_gap_hedge_by_year_state.csv`: summary split by up/down index-year.",
        "- `resset_high_gap_hedge_by_calendar_block.csv`: summary split by broad calendar blocks.",
        "- `resset_high_gap_hedge_by_period_report.md`: this report.",
        "",
        "## Notes",
        "",
        "- The results are gross returns; trading costs, shorting costs, borrow limits, impact, turnover limits, and tax are not deducted.",
        "- A positive `remove_high_active_vs_resset` means removing high Gap stocks improved the index return in that period.",
        "- A positive `short_high_overlay_active` means the high Gap basket underperformed the benchmark in that period.",
    ]
    (output_dir / "resset_high_gap_hedge_by_period_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = HedgeByPeriodConfig(
        high_quantile=float(args.high_quantile),
        hedge_notional=float(args.hedge_notional),
        max_signal_age_days=int(args.max_signal_age_days),
        min_signal_weight_coverage=float(args.min_signal_weight_coverage),
        min_fresh_names=int(args.min_fresh_names),
        min_industry_group_n=int(args.min_industry_group_n),
    )

    index_filter = parse_index_codes(args.index_codes)
    benchmark = read_resset_benchmark(Path(args.index_return_path), index_filter)
    if benchmark.empty:
        raise ValueError("No benchmark rows matched the requested index filter.")
    index_codes = set(benchmark["index_code"].unique())
    benchmark_start = benchmark["trade_date"].min()
    benchmark_end = benchmark["trade_date"].max()
    start_date = pd.to_datetime(args.start_date) if args.start_date else benchmark_start
    end_date = pd.to_datetime(args.end_date) if args.end_date else benchmark_end
    start_date = max(start_date, benchmark_start)
    end_date = min(end_date, benchmark_end)

    benchmark = benchmark[
        (benchmark["trade_date"] >= start_date) & (benchmark["trade_date"] <= end_date)
    ].copy()
    trading_dates = np.sort(benchmark["trade_date"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    signals = read_gap_signals(Path(args.gap_path), Path(args.report_path), trading_dates)
    if not signals.empty:
        first_signal = signals["available_trade_date"].min()
        if pd.notna(first_signal) and first_signal > start_date:
            start_date = first_signal
            benchmark = benchmark[benchmark["trade_date"] >= start_date].copy()

    components = read_components(Path(args.component_path), index_codes)
    components = components[
        (components["end_date_filled"] >= start_date) & (components["begin_date"] <= end_date)
    ].copy()
    stock_ids = sorted(components["stock_id"].dropna().unique().tolist())
    industry = read_industry(Path(args.rawdata_root), stock_ids)
    parquet_paths = list_market_parquets(Path(args.market_daily_root), start_date, end_date)

    daily = build_daily_hedge_returns(
        components=components,
        benchmark=benchmark,
        signals=signals,
        industry=industry,
        parquet_paths=parquet_paths,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    if daily.empty:
        raise ValueError("No daily hedge observations survived the filters.")

    daily = add_period_market_state(daily, "year")
    daily = add_period_market_state(daily, "quarter")

    return_cols = {
        "remove_high_active_vs_resset": "remove_high_active_vs_resset",
        "remove_high_active_vs_recomputed": "remove_high_active_vs_recomputed",
        "short_high_overlay_active": "short_high_overlay_active",
        "high_gap_excess_vs_resset": "high_gap_excess_vs_resset",
    }
    overall = summarize_by(daily, ["index_code", "selection_method"], return_cols)
    by_year = summarize_by(daily, ["index_code", "selection_method", "year"], return_cols)
    by_quarter = summarize_by(daily, ["index_code", "selection_method", "quarter"], return_cols)
    by_year_state = summarize_cross_index_by_date(
        daily,
        ["selection_method", "year_market_state"],
        return_cols,
    )
    by_quarter_state = summarize_cross_index_by_date(
        daily,
        ["selection_method", "quarter_market_state"],
        return_cols,
    )
    by_daily_state = summarize_cross_index_by_date(
        daily,
        ["selection_method", "daily_market_state"],
        return_cols,
    )
    by_block = summarize_cross_index_by_date(daily, ["selection_method", "calendar_block"], return_cols)

    daily.to_csv(
        output_dir / "resset_high_gap_hedge_daily_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        output_dir / "resset_high_gap_hedge_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_year.to_csv(
        output_dir / "resset_high_gap_hedge_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_quarter.to_csv(
        output_dir / "resset_high_gap_hedge_by_quarter.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_year_state.to_csv(
        output_dir / "resset_high_gap_hedge_by_year_state.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_quarter_state.to_csv(
        output_dir / "resset_high_gap_hedge_by_quarter_state.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_daily_state.to_csv(
        output_dir / "resset_high_gap_hedge_by_daily_state.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_block.to_csv(
        output_dir / "resset_high_gap_hedge_by_calendar_block.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(output_dir, daily, overall, by_year, by_year_state, by_block, by_quarter, config)

    print("RESSET high Gap hedge by period completed.")
    print(f"Index codes: {', '.join(sorted(daily['index_code'].unique()))}")
    print(f"Date range: {daily['trade_date'].min().date()} to {daily['trade_date'].max().date()}")
    print(f"Daily rows: {len(daily)}")
    print(f"Output dir: {output_dir}")
    main = overall[overall["strategy_metric"].eq("remove_high_active_vs_resset")]
    print(main.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove or short high Gap stocks inside RESSET broad-index constituents by period."
    )
    parser.add_argument("--gap-path", default=str(DEFAULT_GAP_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPT_PATH))
    parser.add_argument("--rawdata-root", default=str(DEFAULT_RAWDATA_ROOT))
    parser.add_argument("--index-return-path", default=str(DEFAULT_INDEX_RETURN_PATH))
    parser.add_argument("--component-path", default=str(DEFAULT_COMPONENT_PATH))
    parser.add_argument("--market-daily-root", default=str(DEFAULT_MARKET_DAILY_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--index-codes", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--high-quantile", type=float, default=0.20)
    parser.add_argument("--hedge-notional", type=float, default=0.25)
    parser.add_argument("--max-signal-age-days", type=int, default=540)
    parser.add_argument("--min-signal-weight-coverage", type=float, default=0.50)
    parser.add_argument("--min-fresh-names", type=int, default=30)
    parser.add_argument("--min-industry-group-n", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
