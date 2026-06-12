from __future__ import annotations

import argparse
import json
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from build_hs300_gap_state_recognizer import PROJECT_ROOT
from control_hs300_gap_state_turnover import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_COMPONENT_PATH,
    DEFAULT_GAP_PATH,
    DEFAULT_MARKET_DAILY_ROOT,
    DEFAULT_REPT_PATH,
    build_raw_stock_panel,
)
from run_core_cs500_gap_state_strategy import (
    DEFAULT_HMM_PATH,
    INDEX_CODE,
    add_curves,
    build_core_lines,
    build_date_infos,
    plot_core,
    read_market_state,
    summarize,
    write_report,
)


DEFAULT_EXPANDING_QUARTERLY_HMM = (
    PROJECT_ROOT
    / "market_state_model"
    / "output"
    / "feature_tuned_expanding_hmm"
    / "4state"
    / "market_state_probabilities_4state_feature_tuned_expanding_quarterly.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "qlib_framework" / "output" / "core_cs500_time_hmm_robustness"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    hmm_label: str
    hmm_path: Path
    start_date: str
    end_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CS500 core strategy time-window and HMM-refit robustness.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixed-hmm-path", type=Path, default=DEFAULT_HMM_PATH)
    parser.add_argument("--expanding-quarterly-hmm-path", type=Path, default=DEFAULT_EXPANDING_QUARTERLY_HMM)
    parser.add_argument("--current-start-date", default="2021-06-02")
    parser.add_argument("--extended-start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--cost-rate", type=float, default=0.0005)
    parser.add_argument("--high-quantile", type=float, default=0.20)
    parser.add_argument(
        "--stock-risk-measure",
        choices=["leverage_gap", "prev_quarter_volatility", "prev_quarter_max_drawdown"],
        default="leverage_gap",
    )
    parser.add_argument(
        "--gap-definition",
        choices=["rank_signed", "rank_signed_raw", "rank_abs", "hard_signed", "hard_abs"],
        default="rank_signed",
    )
    parser.add_argument("--hard-gap-threshold", type=float, default=0.10)
    parser.add_argument("--max-signal-age-days", type=int, default=540)
    parser.add_argument("--min-fresh-names", type=int, default=30)
    parser.add_argument("--min-hold-days", type=int, default=5)
    return parser.parse_args()


def case_args(args: argparse.Namespace, case: CaseSpec) -> Namespace:
    return Namespace(
        index_code=INDEX_CODE,
        start_date=case.start_date,
        end_date=case.end_date,
        cost_rate=args.cost_rate,
        high_quantile=args.high_quantile,
        gap_definition=args.gap_definition,
        hard_gap_threshold=args.hard_gap_threshold,
        max_signal_age_days=args.max_signal_age_days,
        min_fresh_names=args.min_fresh_names,
        min_hold_days=args.min_hold_days,
        hmm_path=case.hmm_path,
        stock_risk_measure=args.stock_risk_measure,
        output_dir=args.output_dir / case.name,
        gap_path=DEFAULT_GAP_PATH,
        report_path=DEFAULT_REPT_PATH,
        component_path=DEFAULT_COMPONENT_PATH,
        benchmark_path=DEFAULT_BENCHMARK_PATH,
        market_daily_root=DEFAULT_MARKET_DAILY_ROOT,
    )


def strategy_args(args: Namespace) -> Namespace:
    return Namespace(
        index_code=args.index_code,
        benchmark_path=args.benchmark_path,
        component_path=args.component_path,
        market_daily_root=args.market_daily_root,
        gap_path=args.gap_path,
        report_path=args.report_path,
        max_signal_age_days=args.max_signal_age_days,
        min_fresh_names=args.min_fresh_names,
        high_quantile=args.high_quantile,
        stock_risk_measure=getattr(args, "stock_risk_measure", "leverage_gap"),
        gap_definition=args.gap_definition,
        hard_gap_threshold=args.hard_gap_threshold,
    )


def build_cases(args: argparse.Namespace) -> list[CaseSpec]:
    return [
        CaseSpec(
            name="fixed_hmm_current_window",
            hmm_label="fixed_full_sample_hmm",
            hmm_path=args.fixed_hmm_path,
            start_date=args.current_start_date,
            end_date=args.end_date,
        ),
        CaseSpec(
            name="expanding_quarterly_hmm_current_window",
            hmm_label="expanding_quarterly_hmm",
            hmm_path=args.expanding_quarterly_hmm_path,
            start_date=args.current_start_date,
            end_date=args.end_date,
        ),
        CaseSpec(
            name="fixed_hmm_extended_2019",
            hmm_label="fixed_full_sample_hmm",
            hmm_path=args.fixed_hmm_path,
            start_date=args.extended_start_date,
            end_date=args.end_date,
        ),
        CaseSpec(
            name="expanding_quarterly_hmm_extended_2019",
            hmm_label="expanding_quarterly_hmm",
            hmm_path=args.expanding_quarterly_hmm_path,
            start_date=args.extended_start_date,
            end_date=args.end_date,
        ),
    ]


def run_case(args: argparse.Namespace, case: CaseSpec) -> pd.DataFrame:
    run_args = case_args(args, case)
    run_args.output_dir.mkdir(parents=True, exist_ok=True)

    state = read_market_state(run_args.hmm_path, run_args.start_date, run_args.end_date)
    panel = build_raw_stock_panel(strategy_args(run_args), state[["trade_date", "market_regime"]].copy())
    date_infos = build_date_infos(panel)
    daily = build_core_lines(date_infos, state, run_args)
    summary = summarize(daily)
    curves = add_curves(daily)
    plot_path = plot_core(curves, run_args.output_dir)
    report_path = write_report(run_args, summary, curves, plot_path, run_args.output_dir)

    daily.to_parquet(run_args.output_dir / "core_cs500_gap_state_strategy_daily.parquet", index=False)
    daily.to_csv(run_args.output_dir / "core_cs500_gap_state_strategy_daily.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(run_args.output_dir / "core_cs500_gap_state_strategy_curves.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run_args.output_dir / "core_cs500_gap_state_strategy_summary.csv", index=False, encoding="utf-8-sig")

    state_counts = state["market_regime"].value_counts().to_dict()
    core_daily = daily.loc[daily["variant"].eq("gap_state_core")].copy()
    leg_counts = core_daily["selected_leg"].value_counts().to_dict()
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_name": case.name,
        "hmm_label": case.hmm_label,
        "hmm_path": str(case.hmm_path),
        "start_date": case.start_date,
        "end_date": case.end_date,
        "report_path": str(report_path),
        "market_regime_counts": state_counts,
        "core_leg_counts": leg_counts,
    }
    (run_args.output_dir / "core_cs500_time_hmm_case_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out = summary.copy()
    out.insert(0, "case_name", case.name)
    out.insert(1, "hmm_label", case.hmm_label)
    out.insert(2, "start_date", case.start_date)
    out.insert(3, "end_date", case.end_date)
    out["market_regime_counts"] = json.dumps(state_counts, ensure_ascii=False, sort_keys=True)
    out["core_leg_counts"] = json.dumps(leg_counts, ensure_ascii=False, sort_keys=True)
    out["output_dir"] = str(run_args.output_dir)
    return out


def write_summary_report(summary: pd.DataFrame, output_dir: Path) -> Path:
    core = summary.loc[summary["variant"].eq("gap_state_core")].copy()
    keep_cols = [
        "case_name",
        "hmm_label",
        "start_date",
        "end_date",
        "n_days",
        "cum_return_net",
        "excess_vs_index",
        "active_ir",
        "max_drawdown",
        "cash_days",
        "high_gap_days",
        "sum_cost",
    ]
    lines = [
        "# CS500 Time and HMM-Refit Robustness",
        "",
        "## Core Line",
        "",
        core[keep_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Notes",
        "",
        "- fixed_full_sample_hmm uses the locked feature-tuned 4-state HMM file.",
        "- expanding_quarterly_hmm refits the same feature-tuned 4-state HMM before each quarter using only earlier observations.",
        "- Strategy, Gap source, PIT report-date logic, cost, high-Gap bucket, and risk gate are held fixed.",
    ]
    path = output_dir / "core_cs500_time_hmm_robustness_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)
    missing = [case.hmm_path for case in cases if not case.hmm_path.exists()]
    if missing:
        raise FileNotFoundError("Missing HMM probability file(s): " + ", ".join(str(path) for path in missing))

    rows = []
    for case in cases:
        print(f"Running {case.name}: {case.start_date} to {case.end_date}, {case.hmm_label}")
        rows.append(run_case(args, case))
    summary = pd.concat(rows, ignore_index=True)
    summary_path = args.output_dir / "core_cs500_time_hmm_robustness_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path = write_summary_report(summary, args.output_dir)
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
