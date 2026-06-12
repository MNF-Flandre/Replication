from __future__ import annotations

import argparse
import math
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENT_DIR = PROJECT_ROOT / "etf_weight"
DEFAULT_CODES_FILE = DEFAULT_COMPONENT_DIR / "broad_based_index_codes_core.txt"
DEFAULT_MARKET_DAILY_ROOT = Path(
    os.environ.get("QUANT_MARKET_DAILY_ROOT", PROJECT_ROOT / "external_data" / "market_daily")
).expanduser()
DEFAULT_OFFICIAL_INDEX_PATH = Path(
    os.environ.get("QUANT_OFFICIAL_INDEX_PATH", PROJECT_ROOT / "external_data" / "index" / "TRD_Index.csv")
).expanduser()
DEFAULT_OUTPUT_DIR = DEFAULT_COMPONENT_DIR / "output" / "resset_index_returns"
TRADING_DAYS = 252


def normalize_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text


def read_index_codes(path: Path) -> list[str]:
    if not path.exists():
        return ["000300", "000905", "000906", "000852", "932000", "000510", "000985"]
    codes: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        code = normalize_code(line.split()[0])
        if code and code not in codes:
            codes.append(code)
    return codes


def read_resset_components(
    component_dir: Path,
    index_codes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    zip_paths = sorted(component_dir.glob("RESSET*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No RESSET*.zip files found under {component_dir}")

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if name.upper().endswith(".CSV") and "RESSET_IDXCOMPO" in name.upper()
            ]
            for csv_name in sorted(csv_names):
                with archive.open(csv_name) as fh:
                    df = pd.read_csv(
                        fh,
                        dtype=str,
                        encoding="gb18030",
                        low_memory=False,
                        usecols=[3, 4, 11, 12, 13, 14, 15],
                    )
                df.columns = [
                    "index_code",
                    "index_name",
                    "stock_id",
                    "stock_name",
                    "begin_date",
                    "end_date",
                    "component_flag",
                ]
                original_rows = len(df)
                df["index_code"] = df["index_code"].map(normalize_code)
                df["stock_id"] = df["stock_id"].map(normalize_code)
                df = df[df["index_code"].isin(index_codes)].copy()
                df["begin_date"] = pd.to_datetime(df["begin_date"], errors="coerce")
                df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
                df = df.dropna(subset=["index_code", "stock_id", "begin_date"]).copy()
                if not df.empty:
                    df["source_zip"] = zip_path.name
                    df["source_file"] = csv_name
                    rows.append(df)
                manifest.append(
                    {
                        "source_zip": zip_path.name,
                        "source_file": csv_name,
                        "raw_rows": int(original_rows),
                        "kept_rows": int(len(df)),
                        "kept_index_codes": int(df["index_code"].nunique()) if not df.empty else 0,
                        "min_begin_date": df["begin_date"].min() if not df.empty else pd.NaT,
                        "max_begin_date": df["begin_date"].max() if not df.empty else pd.NaT,
                    }
                )

    if not rows:
        wanted = ", ".join(sorted(index_codes))
        raise ValueError(f"No RESSET constituent rows matched requested codes: {wanted}")

    components = pd.concat(rows, ignore_index=True)
    components = components.drop_duplicates(
        ["index_code", "stock_id", "begin_date", "end_date"],
        keep="last",
    )
    components["end_date_filled"] = components["end_date"].fillna(pd.Timestamp("2099-12-31"))
    components = components.sort_values(["index_code", "stock_id", "begin_date"]).reset_index(drop=True)
    return components, pd.DataFrame(manifest)


def list_market_parquets(root: Path) -> list[str]:
    paths = sorted(root.glob("year=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return [str(path) for path in paths]


def build_market_lazy(parquet_paths: list[str], stock_ids: list[str]) -> pl.LazyFrame:
    return (
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
        .sort(["stock_id", "trade_date"])
        .with_columns(
            pl.col("daily_market_cap")
            .shift(1)
            .over("stock_id")
            .alias("lag_daily_market_cap")
        )
    )


def compute_one_index_returns(
    components: pd.DataFrame,
    parquet_paths: list[str],
    index_code: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    comp = components[components["index_code"].eq(index_code)].copy()
    if comp.empty:
        return pd.DataFrame()

    index_name = str(comp["index_name"].dropna().iloc[-1]) if comp["index_name"].notna().any() else ""
    stock_ids = sorted(comp["stock_id"].dropna().unique().tolist())
    comp_pl = pl.DataFrame(
        {
            "index_code": comp["index_code"].astype(str).tolist(),
            "index_name": comp["index_name"].fillna("").astype(str).tolist(),
            "stock_id": comp["stock_id"].astype(str).tolist(),
            "begin_date": comp["begin_date"].dt.strftime("%Y-%m-%d").tolist(),
            "end_date_filled": comp["end_date_filled"].dt.strftime("%Y-%m-%d").tolist(),
        }
    ).with_columns(
        [
            pl.col("begin_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col("end_date_filled").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        ]
    )

    joined = (
        build_market_lazy(parquet_paths, stock_ids)
        .join(comp_pl.lazy(), on="stock_id", how="inner")
        .filter(pl.col("trade_date") >= pl.col("begin_date"))
        .filter(pl.col("trade_date") <= pl.col("end_date_filled"))
    )
    if start_date is not None:
        joined = joined.filter(pl.col("trade_date") >= pl.lit(start_date.date()))
    if end_date is not None:
        joined = joined.filter(pl.col("trade_date") <= pl.lit(end_date.date()))

    joined = (
        joined.sort(["trade_date", "stock_id", "begin_date"])
        .unique(["trade_date", "stock_id"], keep="last")
        .with_columns(
            [
                (
                    (pl.col("lag_daily_market_cap") > 0)
                    & pl.col("change_ratio").is_not_null()
                ).alias("valid_vw"),
                (pl.col("trade_status") != 1).alias("non_normal_trade_status"),
            ]
        )
    )

    daily = (
        joined.group_by("trade_date")
        .agg(
            [
                pl.len().alias("n_active_constituents_matched"),
                pl.col("change_ratio").drop_nulls().count().alias("n_return_constituents"),
                pl.col("change_ratio").mean().alias("equal_weight_return"),
                pl.col("valid_vw").sum().alias("n_value_weight_constituents"),
                pl.when(pl.col("valid_vw"))
                .then(pl.col("lag_daily_market_cap"))
                .otherwise(0.0)
                .sum()
                .alias("lag_market_cap_sum_thousand"),
                pl.when(pl.col("valid_vw"))
                .then(pl.col("lag_daily_market_cap") * pl.col("change_ratio"))
                .otherwise(0.0)
                .sum()
                .alias("lag_market_cap_weighted_return_numer"),
                pl.col("daily_market_cap").sum().alias("same_day_market_cap_sum_thousand"),
                pl.col("non_normal_trade_status").sum().alias("n_non_normal_trade_status"),
            ]
        )
        .with_columns(
            pl.when(pl.col("lag_market_cap_sum_thousand") > 0)
            .then(
                pl.col("lag_market_cap_weighted_return_numer")
                / pl.col("lag_market_cap_sum_thousand")
            )
            .otherwise(None)
            .alias("market_cap_weighted_return")
        )
        .with_columns(
            [
                pl.lit(index_code).alias("index_code"),
                pl.lit(index_name).alias("index_name"),
                (
                    pl.col("n_value_weight_constituents")
                    / pl.col("n_return_constituents")
                ).alias("value_weight_constituent_coverage"),
                pl.lit("lagged_daily_circulated_market_cap").alias("value_weight_method"),
            ]
        )
        .select(
            [
                "index_code",
                "index_name",
                "trade_date",
                "equal_weight_return",
                "market_cap_weighted_return",
                "n_active_constituents_matched",
                "n_return_constituents",
                "n_value_weight_constituents",
                "value_weight_constituent_coverage",
                "lag_market_cap_sum_thousand",
                "same_day_market_cap_sum_thousand",
                "n_non_normal_trade_status",
                "value_weight_method",
            ]
        )
        .sort("trade_date")
        .collect()
    )
    return pd.DataFrame(daily.to_dicts())


def annualized_return(ret: pd.Series) -> float:
    r = ret.dropna()
    if r.empty:
        return math.nan
    cum = float((1.0 + r).prod() - 1.0)
    return float((1.0 + cum) ** (TRADING_DAYS / len(r)) - 1.0)


def cumulative_return(ret: pd.Series) -> float:
    r = ret.dropna()
    if r.empty:
        return math.nan
    return float((1.0 + r).prod() - 1.0)


def annualized_vol(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 2:
        return math.nan
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def max_drawdown(ret: pd.Series) -> float:
    r = ret.dropna()
    if r.empty:
        return math.nan
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def summarize_daily_returns(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index_code, g in daily.groupby("index_code", sort=True):
        ew = g["equal_weight_return"]
        vw = g["market_cap_weighted_return"]
        rows.append(
            {
                "index_code": index_code,
                "index_name": g["index_name"].dropna().iloc[-1] if g["index_name"].notna().any() else "",
                "n_days": int(len(g)),
                "start_date": g["trade_date"].min(),
                "end_date": g["trade_date"].max(),
                "avg_active_constituents_matched": float(g["n_active_constituents_matched"].mean()),
                "avg_value_weight_constituents": float(g["n_value_weight_constituents"].mean()),
                "avg_value_weight_constituent_coverage": float(
                    g["value_weight_constituent_coverage"].mean()
                ),
                "equal_weight_cum_return": cumulative_return(ew),
                "market_cap_weighted_cum_return": cumulative_return(vw),
                "equal_weight_ann_return": annualized_return(ew),
                "market_cap_weighted_ann_return": annualized_return(vw),
                "equal_weight_ann_vol": annualized_vol(ew),
                "market_cap_weighted_ann_vol": annualized_vol(vw),
                "equal_weight_max_drawdown": max_drawdown(ew),
                "market_cap_weighted_max_drawdown": max_drawdown(vw),
                "ew_vw_daily_return_corr": float(ew.corr(vw)) if ew.notna().sum() > 1 else math.nan,
                "mean_daily_ew_minus_vw": float((ew - vw).mean()),
            }
        )
    return pd.DataFrame(rows)


def validate_against_official_index_returns(
    daily: pd.DataFrame,
    official_index_path: Path,
) -> pd.DataFrame:
    if not official_index_path.exists():
        return pd.DataFrame(
            columns=[
                "index_code",
                "n_overlap_days",
                "overlap_start_date",
                "overlap_end_date",
                "corr_official_vs_market_cap_weighted",
                "corr_official_vs_equal_weight",
                "mean_market_cap_weighted_minus_official",
                "mean_equal_weight_minus_official",
            ]
        )
    official = pd.read_csv(
        official_index_path,
        usecols=["Indexcd", "Trddt", "Retindex"],
        dtype={"Indexcd": str},
        low_memory=False,
    )
    official["index_code"] = official["Indexcd"].map(normalize_code)
    official["trade_date"] = pd.to_datetime(official["Trddt"], errors="coerce")
    official["official_return"] = pd.to_numeric(official["Retindex"], errors="coerce")
    official = official.dropna(subset=["index_code", "trade_date", "official_return"])

    base = daily.copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for index_code, g in base.groupby("index_code", sort=True):
        merged = g.merge(
            official[official["index_code"].eq(index_code)][["trade_date", "official_return"]],
            on="trade_date",
            how="inner",
        )
        if merged.empty:
            continue
        rows.append(
            {
                "index_code": index_code,
                "n_overlap_days": int(len(merged)),
                "overlap_start_date": merged["trade_date"].min(),
                "overlap_end_date": merged["trade_date"].max(),
                "corr_official_vs_market_cap_weighted": float(
                    merged["official_return"].corr(merged["market_cap_weighted_return"])
                ),
                "corr_official_vs_equal_weight": float(
                    merged["official_return"].corr(merged["equal_weight_return"])
                ),
                "mean_market_cap_weighted_minus_official": float(
                    (merged["market_cap_weighted_return"] - merged["official_return"]).mean()
                ),
                "mean_equal_weight_minus_official": float(
                    (merged["equal_weight_return"] - merged["official_return"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "(empty)"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    return view.to_markdown(index=False)


def write_report(
    output_dir: Path,
    index_codes: list[str],
    components: pd.DataFrame,
    manifest: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    official_validation: pd.DataFrame,
    component_dir: Path,
    market_daily_root: Path,
) -> None:
    component_summary = (
        components.groupby(["index_code", "index_name"], as_index=False)
        .agg(
            interval_rows=("stock_id", "size"),
            unique_stocks=("stock_id", "nunique"),
            min_begin_date=("begin_date", "min"),
            max_begin_date=("begin_date", "max"),
            max_end_date=("end_date", "max"),
        )
        .sort_values("index_code")
    )
    lines = [
        "# RESSET Broad Index Return Construction",
        "",
        "## Scope",
        "",
        "This output constructs daily broad-index benchmark returns from RESSET constituent intervals and CSMAR daily stock returns.",
        "",
        "- Equal-weighted return: simple average of constituent stock daily returns.",
        "- Market-cap-weighted return: lagged daily circulating market-cap weighted average of constituent stock daily returns.",
        "- The lagged market cap avoids using end-of-day market value to weight the same day's return.",
        "",
        "## Inputs",
        "",
        f"- RESSET constituent directory: `{component_dir}`",
        f"- Daily stock return directory: `{market_daily_root}`",
        "- Daily market cap field: `daily_market_cap`, mapped from CSMAR `Dsmvosd`.",
        "- CSMAR field definition: `Dsmvosd = circulating shares * closing price`, unit is thousand currency units.",
        "",
        "## Requested Index Codes",
        "",
        "`" + ", ".join(index_codes) + "`",
        "",
        "## Constituent Coverage",
        "",
        markdown_table(component_summary),
        "",
        "## Return Summary",
        "",
        markdown_table(summary),
        "",
        "## Official Index Sanity Check",
        "",
        "Where official CSMAR index returns are available locally, the constructed benchmark is compared with `Retindex`.",
        "",
        markdown_table(official_validation),
        "",
        "## Source Manifest",
        "",
        markdown_table(manifest[manifest["kept_rows"].gt(0)].sort_values(["source_zip", "source_file"])),
        "",
        "## Output Files",
        "",
        "- `resset_index_daily_returns.csv`: index-date panel with equal-weight and market-cap-weight returns.",
        "- `resset_index_return_summary.csv`: index-level performance and coverage summary.",
        "- `resset_component_intervals.csv`: cleaned RESSET constituent intervals used in the join.",
        "- `resset_component_manifest.csv`: source zip/file inventory.",
        "- `official_index_return_validation.csv`: sanity check against local official index return data when available.",
        "",
        "## Notes",
        "",
        "- Constituent intervals are treated as active when `begin_date <= trade_date <= end_date`; missing end dates are treated as active through the sample end.",
        "- Market-cap weights use prior trading day's `daily_market_cap` within each stock.",
        "- The output is a benchmark construction layer only. It does not apply Gap filtering or enhancement yet.",
    ]
    (output_dir / "resset_index_return_construction_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build equal-weight and market-cap-weighted broad index returns.")
    parser.add_argument("--component-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--codes-file", type=Path, default=DEFAULT_CODES_FILE)
    parser.add_argument("--market-daily-root", type=Path, default=DEFAULT_MARKET_DAILY_ROOT)
    parser.add_argument("--official-index-path", type=Path, default=DEFAULT_OFFICIAL_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    index_codes = read_index_codes(args.codes_file)
    components, manifest = read_resset_components(args.component_dir, set(index_codes))
    parquet_paths = list_market_parquets(args.market_daily_root)
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    daily_parts: list[pd.DataFrame] = []
    for index_code in index_codes:
        part = compute_one_index_returns(components, parquet_paths, index_code, start_date, end_date)
        if part.empty:
            print(f"[WARN] No daily returns constructed for index {index_code}")
            continue
        daily_parts.append(part)
        print(
            f"[OK] {index_code}: {len(part):,} days, "
            f"{part['trade_date'].min()} to {part['trade_date'].max()}, "
            f"avg constituents={part['n_active_constituents_matched'].mean():.1f}"
        )

    if not daily_parts:
        raise RuntimeError("No index return panels were constructed.")

    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily.sort_values(["index_code", "trade_date"]).reset_index(drop=True)
    summary = summarize_daily_returns(daily).sort_values("index_code").reset_index(drop=True)
    official_validation = validate_against_official_index_returns(daily, args.official_index_path)

    components.to_csv(output_dir / "resset_component_intervals.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(output_dir / "resset_component_manifest.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "resset_index_daily_returns.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "resset_index_return_summary.csv", index=False, encoding="utf-8-sig")
    official_validation.to_csv(
        output_dir / "official_index_return_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(
        output_dir,
        index_codes,
        components,
        manifest,
        daily,
        summary,
        official_validation,
        args.component_dir,
        args.market_daily_root,
    )

    print("")
    print(f"Output directory: {output_dir}")
    print(f"Daily returns: {output_dir / 'resset_index_daily_returns.csv'}")
    print(f"Summary: {output_dir / 'resset_index_return_summary.csv'}")
    print(f"Official validation: {output_dir / 'official_index_return_validation.csv'}")
    print(f"Report: {output_dir / 'resset_index_return_construction_report.md'}")


if __name__ == "__main__":
    main()
