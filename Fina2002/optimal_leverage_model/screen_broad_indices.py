from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from gap_index_enhancement import (
    DEFAULT_ETF_WEIGHT_DIR,
    BROAD_INDEX_NAME_MAP,
    markdown_table,
    read_index_weights,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "broad_index_screen"

STRICT_TARGETS = {
    "000300": "CSI 300 / 沪深300",
    "399300": "CSI 300 / 沪深300",
    "000905": "CSI 500 / 中证500",
    "399905": "CSI 500 / 中证500",
    "000906": "CSI 800 / 中证800",
    "399906": "CSI 800 / 中证800",
    "000852": "CSI 1000 / 中证1000",
    "399852": "CSI 1000 / 中证1000",
}

BROAD_SUPPLEMENTS = {
    "000010": "SSE 180 / 上证180",
    "399903": "CSI 100 historical venue code / 中证100候选",
    "399904": "CSI 200 historical venue code / 中证200候选",
}

NOT_SELECTED_REASON = {
    "000019": "not in strict broad-index target list; index identity not confirmed from local files",
    "000021": "not in strict broad-index target list; index identity not confirmed from local files",
    "000160": "not in strict broad-index target list; index identity not confirmed from local files",
    "930649": "not a requested broad index code; likely thematic/style index from local code pattern",
    "930660": "not a requested broad index code; likely thematic/style index from local code pattern",
    "930661": "not a requested broad index code; likely thematic/style index from local code pattern",
}


def index_summary(weights: pd.DataFrame) -> pd.DataFrame:
    return (
        weights.groupby("index_code")
        .agg(
            rows=("firm_id", "size"),
            dates=("weight_date", "nunique"),
            min_date=("weight_date", "min"),
            max_date=("weight_date", "max"),
            avg_constituents=("firm_id", lambda s: len(s) / weights.loc[s.index, "weight_date"].nunique()),
            stock_ids=("firm_id", "nunique"),
            weight_sum_median=("index_weight", lambda s: 100.0 * s.groupby(weights.loc[s.index, "weight_date"]).sum().median()),
        )
        .reset_index()
        .sort_values(["dates", "rows"], ascending=[False, False])
    )


def build_screen(summary: pd.DataFrame) -> pd.DataFrame:
    available = set(summary["index_code"])
    wanted = {**STRICT_TARGETS, **BROAD_SUPPLEMENTS}
    rows: list[dict[str, Any]] = []
    for code, name in wanted.items():
        base = summary[summary["index_code"].eq(code)]
        row: dict[str, Any] = {
            "index_code": code,
            "index_name_expected": name,
            "target_group": "strict_requested" if code in STRICT_TARGETS else "broad_supplement",
            "available_in_etf_weight": code in available,
            "selected_for_main_tests": False,
            "selection_reason": "",
        }
        if not base.empty:
            row.update(base.iloc[0].to_dict())
            if code in STRICT_TARGETS:
                row["selected_for_main_tests"] = True
                row["selection_reason"] = "available strict requested broad index"
            elif code == "000010":
                row["selected_for_main_tests"] = True
                row["selection_reason"] = "available broad supplement: SSE 180"
            else:
                row["selection_reason"] = (
                    "available broad supplement but ends before A_clean_main Gap coverage; keep out of main tests"
                )
        else:
            row["selection_reason"] = "not found in current etf_weight files"
        rows.append(row)
    return pd.DataFrame(rows)


def build_excluded(summary: pd.DataFrame, selected_codes: set[str]) -> pd.DataFrame:
    rows = []
    for item in summary.itertuples(index=False):
        code = str(item.index_code)
        if code in selected_codes:
            continue
        rows.append(
            {
                "index_code": code,
                "rows": int(item.rows),
                "dates": int(item.dates),
                "min_date": item.min_date,
                "max_date": item.max_date,
                "avg_constituents": float(item.avg_constituents),
                "reason": NOT_SELECTED_REASON.get(
                    code,
                    "not in strict requested broad-index list and not selected as broad supplement",
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, summary: pd.DataFrame, screen: pd.DataFrame, excluded: pd.DataFrame) -> None:
    selected = screen[screen["selected_for_main_tests"]].copy()
    selected_codes = ",".join(selected["index_code"].tolist())
    strict_available = screen[
        screen["target_group"].eq("strict_requested") & screen["available_in_etf_weight"]
    ].copy()
    lines = [
        "# 宽基指数筛选报告",
        "",
        "## 结论",
        "",
        f"- 当前 `etf_weight` 中严格匹配到的目标宽基指数只有：`{','.join(strict_available['index_code'].tolist()) or 'none'}`。",
        f"- 建议当前主检验指数池：`{selected_codes}`。",
        "- 其中 `000906` 是严格目标池中的中证800；`000010` 是上证180，作为宽基补充。",
        "- 沪深300、 中证500、 中证1000 当前未在本地 `etf_weight` 文件中出现。",
        "- 中证1500没有在本地文件中以可识别代码出现；如要纳入，需要补充对应指数权重文件或明确供应商代码。",
        "- `399903/399904` 虽然像中证100/200历史代码，但权重截止到2012/2014，早于 A_clean_main 的主要 Gap 覆盖期，因此不进入主回测。",
        "",
        "## 目标宽基筛选表",
        "",
        markdown_table(screen),
        "",
        "## 本地全部指数概览",
        "",
        markdown_table(summary),
        "",
        "## 排除指数",
        "",
        markdown_table(excluded),
        "",
        "## 后续运行建议",
        "",
        "高 Gap 对冲检验只跑当前主池：",
        "",
        "```powershell",
        f"python optimal_leverage_model/gap_high_hedge_test.py --index-codes {selected_codes}",
        "```",
        "",
        "指数增强倾斜检验只跑当前主池：",
        "",
        "```powershell",
        f"python optimal_leverage_model/gap_index_enhancement.py --index-codes {selected_codes}",
        "```",
    ]
    (output_dir / "broad_index_screen_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights, manifest = read_index_weights(Path(args.etf_weight_dir))
    summary = index_summary(weights)
    screen = build_screen(summary)
    selected_codes = set(screen.loc[screen["selected_for_main_tests"], "index_code"])
    excluded = build_excluded(summary, selected_codes)

    manifest.to_csv(output_dir / "broad_index_source_manifest.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "local_index_inventory.csv", index=False, encoding="utf-8-sig")
    screen.to_csv(output_dir / "broad_index_selection.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(output_dir / "excluded_index_inventory.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, summary, screen, excluded)

    print("Broad index screen completed.")
    print("Selected codes:", ",".join(sorted(selected_codes)))
    print(f"Report: {output_dir / 'broad_index_screen_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen broad indices from local index weight files.")
    parser.add_argument("--etf-weight-dir", default=str(DEFAULT_ETF_WEIGHT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
