from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from gap_index_enhancement import (
    DEFAULT_ETF_WEIGHT_DIR,
    DEFAULT_OUTPUT_DIR,
    normalize_index_code,
    read_index_weights,
)
from validate_optimal_leverage import markdown_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_INFO_GLOB = "指数基本信息文件*.zip"
DEFAULT_SCREEN_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "index_universe"

BROAD_INDEX_TARGETS = {
    "000300": "CSI300 / HS300",
    "399300": "CSI300 / HS300",
    "000905": "CSI500",
    "399905": "CSI500",
    "000906": "CSI800",
    "399906": "CSI800",
    "000852": "CSI1000",
    "399852": "CSI1000",
    "932000": "CSI2000",
    "000510": "CSI A500",
    "000985": "CSI All Share",
    "399985": "CSI All Share",
    "000010": "SSE180",
    "000016": "SSE50",
    "000903": "CSI A100",
    "399903": "CSI A100",
    "399904": "CSI Midcap 200",
}

CORE_RESEARCH_CODES = {
    "000300",
    "399300",
    "000905",
    "399905",
    "000906",
    "399906",
    "000852",
    "399852",
    "932000",
    "000510",
}


def read_index_info(etf_weight_dir: Path) -> pd.DataFrame:
    matches = sorted(etf_weight_dir.glob(DEFAULT_INDEX_INFO_GLOB))
    if not matches:
        raise FileNotFoundError(f"No index info zip found under {etf_weight_dir}")
    path = matches[0]
    with zipfile.ZipFile(path) as archive:
        with archive.open("IDX_Idxinfo.csv") as fh:
            info = pd.read_csv(
                fh,
                usecols=["Indexcd", "Idxinfo01", "Idxinfo11", "Idxinfo08", "Idxinfo09"],
                dtype={"Indexcd": "string"},
                encoding="utf-8-sig",
                low_memory=False,
            )
    info["index_code"] = info["Indexcd"].map(normalize_index_code)
    info = info.rename(
        columns={
            "Indexcd": "raw_index_code",
            "Idxinfo01": "index_name",
            "Idxinfo11": "index_start_date",
            "Idxinfo08": "index_type",
            "Idxinfo09": "market_type",
        }
    )
    return info


def primary_index_info(info: pd.DataFrame) -> pd.DataFrame:
    out = info.copy()
    raw = out["raw_index_code"].astype(str).str.strip()
    exact = raw.str.fullmatch(r"\d{6}")
    primary = out[exact & raw.eq(out["index_code"])].copy()
    primary = primary.sort_values(["index_code", "index_start_date"]).drop_duplicates("index_code", keep="first")
    return primary


def summarize_weights(etf_weight_dir: Path) -> pd.DataFrame:
    weights, _ = read_index_weights(etf_weight_dir)
    return (
        weights.groupby("index_code", as_index=False)
        .agg(
            weight_rows=("firm_id", "size"),
            weight_dates=("weight_date", "nunique"),
            min_weight_date=("weight_date", "min"),
            max_weight_date=("weight_date", "max"),
            stock_ids=("firm_id", "nunique"),
            avg_constituents=("firm_id", lambda s: len(s) / weights.loc[s.index, "weight_date"].nunique()),
        )
        .sort_values(["weight_dates", "weight_rows"], ascending=False)
    )


def summarize_gap_coverage(index_daily_path: Path) -> pd.DataFrame:
    if not index_daily_path.exists():
        return pd.DataFrame(
            columns=[
                "index_code",
                "gap_test_days",
                "avg_signal_weight_coverage",
                "days_signal_coverage_ge_50pct",
                "days_signal_coverage_ge_80pct",
            ]
        )
    daily = pd.read_csv(
        index_daily_path,
        usecols=["index_code", "weight_date", "tilt_strength", "strategy_direction", "signal_weight_coverage"],
    )
    daily["index_code"] = daily["index_code"].map(normalize_index_code)
    if "tilt_strength" in daily.columns:
        daily = daily[daily["tilt_strength"].eq(daily["tilt_strength"].min())].copy()
    if "strategy_direction" in daily.columns:
        first_direction = sorted(daily["strategy_direction"].dropna().unique())[0]
        daily = daily[daily["strategy_direction"].eq(first_direction)].copy()
    return (
        daily.groupby("index_code", as_index=False)
        .agg(
            gap_test_days=("weight_date", "nunique"),
            avg_signal_weight_coverage=("signal_weight_coverage", "mean"),
            days_signal_coverage_ge_50pct=("signal_weight_coverage", lambda s: int((s >= 0.50).sum())),
            days_signal_coverage_ge_80pct=("signal_weight_coverage", lambda s: int((s >= 0.80).sum())),
        )
        .sort_values(["days_signal_coverage_ge_50pct", "avg_signal_weight_coverage"], ascending=False)
    )


def build_screen(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    etf_weight_dir = Path(args.etf_weight_dir)
    info = read_index_info(etf_weight_dir)
    info_main = primary_index_info(info)
    weights = summarize_weights(etf_weight_dir)
    gap = summarize_gap_coverage(Path(args.index_daily_path))

    target = pd.DataFrame(
        [{"index_code": code, "broad_label": label} for code, label in BROAD_INDEX_TARGETS.items()]
    )
    target["research_core_requested"] = target["index_code"].isin(CORE_RESEARCH_CODES)
    screen = (
        target.merge(info_main, on="index_code", how="left")
        .merge(weights, on="index_code", how="left")
        .merge(gap, on="index_code", how="left")
    )
    screen["has_index_info"] = screen["index_name"].notna()
    screen["has_weight_file"] = screen["weight_rows"].notna()
    screen["has_gap_coverage_ge_50pct"] = screen["days_signal_coverage_ge_50pct"].fillna(0).gt(0)
    screen["usable_now"] = screen["has_weight_file"] & screen["has_gap_coverage_ge_50pct"]

    available_weights = weights.merge(info_main, on="index_code", how="left")
    available_weights["broad_candidate"] = available_weights["index_code"].isin(BROAD_INDEX_TARGETS)
    return screen.sort_values(["research_core_requested", "usable_now", "has_weight_file", "index_code"], ascending=[False, False, False, True]), available_weights


def write_report(output_dir: Path, screen: pd.DataFrame, available_weights: pd.DataFrame) -> None:
    usable = screen[screen["usable_now"]].copy()
    weighted_no_gap = screen[screen["has_weight_file"] & ~screen["usable_now"]].copy()
    missing_weights = screen[~screen["has_weight_file"]].copy()
    lines = [
        "# Broad Index Universe Screen",
        "",
        "## Decision",
        "",
        "Use only broad-based indices. Theme, governance, industry, and concept indices should be excluded from the main index-enhancement tests.",
        "",
        "With the currently downloaded weight files, the usable broad-based indices with post-disclosure Gap coverage are:",
        "",
        markdown_table(
            usable[
                [
                    "index_code",
                    "broad_label",
                    "index_name",
                    "min_weight_date",
                    "max_weight_date",
                    "weight_dates",
                    "avg_constituents",
                    "days_signal_coverage_ge_50pct",
                    "avg_signal_weight_coverage",
                ]
            ],
            max_rows=30,
        ),
        "",
        "Recommended current whitelist:",
        "",
        "`" + ",".join(usable["index_code"].drop_duplicates().tolist()) + "`",
        "",
        "## Requested Broad Indices Missing Weights",
        "",
        "The index basic-info file contains mappings for broad indices such as CSI300, CSI500, CSI1000, CSI2000, and CSI A500, but the current `IDX_Smprat` weight zips do not contain their constituent weights.",
        "",
        markdown_table(
            missing_weights[
                [
                    "index_code",
                    "broad_label",
                    "index_name",
                    "index_start_date",
                    "research_core_requested",
                    "has_index_info",
                    "has_weight_file",
                ]
            ],
            max_rows=50,
        ),
        "",
        "## Broad Indices With Weights But No Gap Coverage",
        "",
        markdown_table(
            weighted_no_gap[
                [
                    "index_code",
                    "broad_label",
                    "index_name",
                    "min_weight_date",
                    "max_weight_date",
                    "weight_dates",
                    "avg_constituents",
                    "days_signal_coverage_ge_50pct",
                ]
            ],
            max_rows=50,
        ),
        "",
        "## All Weight-File Indices",
        "",
        markdown_table(
            available_weights[
                [
                    "index_code",
                    "index_name",
                    "min_weight_date",
                    "max_weight_date",
                    "weight_dates",
                    "avg_constituents",
                    "broad_candidate",
                ]
            ],
            max_rows=50,
        ),
        "",
        "## Implication",
        "",
        "For the current data package, main broad-index tests should use `000906` and `000010`. `399903` and `399904` are broad-like historical CSI size indices, but their weight samples end before the current A-clean Gap signal has usable coverage.",
        "",
        "To test CSI300, CSI500, CSI1000, CSI1500/CSI2000, or CSI A500 formally, their `IDX_Smprat` constituent-weight files must be added to `etf_weight`.",
    ]
    (output_dir / "broad_index_universe_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screen, available_weights = build_screen(args)
    screen.to_csv(output_dir / "broad_index_universe_screen.csv", index=False, encoding="utf-8-sig")
    available_weights.to_csv(output_dir / "available_index_weight_coverage.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, screen, available_weights)
    usable = screen[screen["usable_now"]]["index_code"].drop_duplicates().tolist()
    print("Broad index universe screen completed.")
    print(f"Usable broad whitelist: {','.join(usable)}")
    print(f"Report: {output_dir / 'broad_index_universe_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen broad index candidates from RESSET index info and weight files.")
    parser.add_argument("--etf-weight-dir", default=str(PROJECT_ROOT / "etf_weight"))
    parser.add_argument("--index-daily-path", default=str(DEFAULT_OUTPUT_DIR / "gap_index_enhancement_daily_returns.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_SCREEN_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
