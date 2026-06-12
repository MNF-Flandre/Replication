from __future__ import annotations

import argparse
import json
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from control_hs300_gap_state_turnover import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_COMPONENT_PATH,
    DEFAULT_GAP_PATH,
    DEFAULT_MARKET_DAILY_ROOT,
    DEFAULT_REPT_PATH,
    normalize_code,
)
from run_core_cs500_gap_state_strategy import (
    DEFAULT_HMM_PATH,
    INDEX_CODE,
    PROJECT_ROOT,
    TRADING_DAYS,
    add_curves,
    annualized_return,
    annualized_vol,
    build_date_infos,
    build_raw_stock_panel,
    cumulative_return,
    generate_risk_gate,
    information_ratio,
    max_drawdown,
    read_market_state,
    simulate_leg_sequence,
    strategy_args,
    summarize,
    win_rate,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "qlib_framework" / "output" / "core_cs500_gap_state_robustness"

HMM_VARIANTS = {
    "locked_expanding_4state": DEFAULT_HMM_PATH,
}

INDEX_NAMES = {
    "000300": "沪深300",
    "000852": "中证1000",
    "000905": "中证500",
    "000906": "中证800",
    "000985": "中证全指",
}

FULL_SIGNAL_AGE_DAYS = 36500


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    family: str
    label: str
    index_code: str = INDEX_CODE
    start_date: str = "2021-06-02"
    end_date: str = "2024-12-31"
    cost_rate: float = 0.0005
    high_quantile: float = 0.20
    max_signal_age_days: int = 540
    min_fresh_names: int = 30
    min_hold_days: int = 5
    hmm_name: str = "locked_expanding_4state"
    risk_off_leg: str = "cash"
    gap_refresh: str = "daily"
    stock_risk_measure: str = "leverage_gap"
    gap_definition: str = "rank_signed"
    hard_gap_threshold: float = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-axis robustness checks for the locked CS500 Gap x Market State strategy.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gap-path", type=Path, default=DEFAULT_GAP_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPT_PATH)
    parser.add_argument("--component-path", type=Path, default=DEFAULT_COMPONENT_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--market-daily-root", type=Path, default=DEFAULT_MARKET_DAILY_ROOT)
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional subset: locked_base cost high_quantile stock_risk_measure gap_definition signal_age min_hold risk_off_leg index_universe subperiod gap_refresh",
    )
    return parser.parse_args()


def build_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = [
        CaseSpec("base", "locked_base", "locked CS500 core"),
    ]

    for cost in [0.0, 0.0003, 0.0005, 0.0010]:
        cases.append(CaseSpec(f"cost_{int(cost * 10000):04d}bp", "cost", f"cost={cost:.2%}", cost_rate=cost))

    for quantile in [0.10, 0.20, 0.30]:
        cases.append(
            CaseSpec(
                f"high_gap_top_{int(quantile * 100):02d}",
                "high_quantile",
                f"high gap top {quantile:.0%}",
                high_quantile=quantile,
            )
        )

    for measure, label in [
        ("leverage_gap", "leverage gap"),
        ("prev_quarter_volatility", "previous-quarter volatility"),
        ("prev_quarter_max_drawdown", "previous-quarter max drawdown"),
    ]:
        cases.append(
            CaseSpec(
                f"stock_risk_{measure}",
                "stock_risk_measure",
                label,
                stock_risk_measure=measure,
            )
        )

    for threshold in [0.05, 0.10, 0.15, 0.20]:
        cases.append(
            CaseSpec(
                f"hard_signed_{int(threshold * 100):02d}pp",
                "gap_definition",
                f"d_obs - dstar >= {threshold:.0%}",
                gap_definition="hard_signed",
                hard_gap_threshold=threshold,
            )
        )
    cases.extend(
        [
            CaseSpec(
                "hard_abs_10pp",
                "gap_definition",
                "|d_obs - dstar| >= 10%",
                gap_definition="hard_abs",
                hard_gap_threshold=0.10,
            ),
            CaseSpec(
                "rank_abs_top_20",
                "gap_definition",
                "abs gap top 20%",
                gap_definition="rank_abs",
            ),
        ]
    )

    for age, label in [(365, "365d"), (540, "540d"), (FULL_SIGNAL_AGE_DAYS, "full")]:
        cases.append(
            CaseSpec(
                f"signal_age_{label}",
                "signal_age",
                f"PIT max age {label}",
                max_signal_age_days=age,
            )
        )

    for hold in [3, 5, 10, 20]:
        cases.append(CaseSpec(f"min_hold_{hold:02d}", "min_hold", f"min hold {hold}d", min_hold_days=hold))

    for leg in ["cash", "no_high"]:
        cases.append(
            CaseSpec(
                f"risk_off_{leg}",
                "risk_off_leg",
                f"risk-off {leg}",
                risk_off_leg=leg,
            )
        )

    for code in ["000300", "000905", "000852", "000906", "000985"]:
        cases.append(
            CaseSpec(
                f"index_{code}",
                "index_universe",
                f"{code} {INDEX_NAMES.get(code, '')}".strip(),
                index_code=code,
            )
        )

    for label, start, end in [
        ("2021", "2021-06-02", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
    ]:
        cases.append(
            CaseSpec(
                f"subperiod_{label}",
                "subperiod",
                label,
                start_date=start,
                end_date=end,
            )
        )

    for freq in ["daily", "monthly", "quarterly"]:
        cases.append(CaseSpec(f"gap_refresh_{freq}", "gap_refresh", f"Gap refresh {freq}", gap_refresh=freq))

    out: list[CaseSpec] = []
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        out.append(case)
    return out


def args_for_case(case: CaseSpec, run_args: argparse.Namespace) -> Namespace:
    return Namespace(
        index_code=normalize_code(case.index_code),
        start_date=case.start_date,
        end_date=case.end_date,
        cost_rate=case.cost_rate,
        high_quantile=case.high_quantile,
        gap_definition=case.gap_definition,
        hard_gap_threshold=case.hard_gap_threshold,
        max_signal_age_days=case.max_signal_age_days,
        min_fresh_names=case.min_fresh_names,
        min_hold_days=case.min_hold_days,
        hmm_path=HMM_VARIANTS[case.hmm_name],
        stock_risk_measure=case.stock_risk_measure,
        output_dir=run_args.output_dir,
        gap_path=run_args.gap_path,
        report_path=run_args.report_path,
        component_path=run_args.component_path,
        benchmark_path=run_args.benchmark_path,
        market_daily_root=run_args.market_daily_root,
    )


def refresh_mask(dates: pd.Index, freq: str) -> pd.Series:
    if freq == "daily":
        return pd.Series(True, index=dates)
    s = pd.Series(pd.to_datetime(dates), index=dates)
    if freq == "monthly":
        return s.dt.to_period("M").ne(s.shift(1).dt.to_period("M"))
    if freq == "quarterly":
        return s.dt.to_period("Q").ne(s.shift(1).dt.to_period("Q"))
    raise ValueError(f"Unknown gap refresh frequency: {freq}")


def simulate_core_variant(
    date_infos: list[dict],
    state: pd.DataFrame,
    args: Namespace,
    risk_off_leg: str,
    gap_refresh: str,
) -> pd.DataFrame:
    dates = [info["trade_date"] for info in date_infos]
    index = pd.Index(dates, name="trade_date")
    risk_on = generate_risk_gate(state.loc[state["trade_date"].isin(index)].copy(), args.min_hold_days).reindex(index)
    if risk_on.isna().any():
        raise ValueError("Risk gate is missing dates after aligning market state with the selected trading days.")

    if gap_refresh == "daily":
        legs = pd.Series(np.where(risk_on, "high_gap", risk_off_leg), index=index)
        return simulate_leg_sequence(
            date_infos,
            legs,
            variant="gap_state_core",
            display_name="高Gap+择时",
            cost_rate=args.cost_rate,
        ).merge(risk_on.rename("risk_on").reset_index(), on="trade_date", how="left")

    schedule = refresh_mask(index, gap_refresh)
    previous: dict[str, float] = {}
    previous_leg: str | None = None
    rows = []
    active_gap_targets: dict[str, dict[str, float]] = {}
    leg_map = {pd.Timestamp(k): str(v) for k, v in pd.Series(np.where(risk_on, "high_gap", risk_off_leg), index=index).items()}

    for info in date_infos:
        date = info["trade_date"]
        leg = leg_map.get(date, "high_gap")
        targets = info["targets"]
        if leg in {"high_gap", "no_high"}:
            if bool(schedule.loc[date]) or leg not in active_gap_targets:
                active_gap_targets[leg] = targets[leg].copy()
            if previous_leg == leg and not bool(schedule.loc[date]):
                target = previous.copy()
            else:
                target = active_gap_targets[leg]
        else:
            target = targets.get(leg, targets["index"])

        keys = set(previous) | set(target)
        turnover = float(sum(abs(target.get(key, 0.0) - previous.get(key, 0.0)) for key in keys))
        gross_ret = float(sum(weight * info["returns"].get(stock_id, 0.0) for stock_id, weight in target.items()))
        cost = args.cost_rate * turnover
        denom = 1.0 + gross_ret
        if target and denom > 0:
            previous = {
                stock_id: float(weight * (1.0 + info["returns"].get(stock_id, 0.0)) / denom)
                for stock_id, weight in target.items()
                if abs(weight) > 1e-12
            }
        else:
            previous = target.copy()
        rows.append(
            {
                "trade_date": date,
                "variant": "gap_state_core",
                "display_name": "高Gap+择时",
                "selected_leg": leg,
                "gross_return": gross_ret,
                "cost": cost,
                "net_return": gross_ret - cost,
                "gross_turnover": turnover,
                "n_names": int(len(target)),
            }
        )
        previous_leg = leg
    return pd.DataFrame(rows).merge(risk_on.rename("risk_on").reset_index(), on="trade_date", how="left")


def summarize_core(case_daily: pd.DataFrame, index_daily: pd.DataFrame) -> dict[str, float | int | str]:
    merged = case_daily.merge(
        index_daily[["trade_date", "net_return"]].rename(columns={"net_return": "index_net_return"}),
        on="trade_date",
        how="left",
    )
    active = merged["net_return"] - merged["index_net_return"]
    high_gap_rows = merged.loc[merged["selected_leg"].eq("high_gap")]
    return {
        "n_days": int(len(merged)),
        "cum_return_net": cumulative_return(merged["net_return"]),
        "cum_return_gross": cumulative_return(merged["gross_return"]),
        "index_cum_return": cumulative_return(merged["index_net_return"]),
        "excess_vs_index": cumulative_return(merged["net_return"]) - cumulative_return(merged["index_net_return"]),
        "ann_return": annualized_return(merged["net_return"]),
        "ann_vol": annualized_vol(merged["net_return"]),
        "active_ir": information_ratio(active),
        "max_drawdown": max_drawdown(merged["net_return"]),
        "win_rate": win_rate(merged["net_return"]),
        "avg_turnover": float(merged["gross_turnover"].mean()),
        "ann_turnover": float(merged["gross_turnover"].mean() * TRADING_DAYS),
        "sum_cost": float(merged["cost"].sum()),
        "cash_days": int(merged["selected_leg"].eq("cash").sum()),
        "no_high_days": int(merged["selected_leg"].eq("no_high").sum()),
        "high_gap_days": int(merged["selected_leg"].eq("high_gap").sum()),
        "index_days": int(merged["selected_leg"].eq("index").sum()),
        "avg_n_names": float(merged["n_names"].mean()),
        "avg_high_gap_n_names": float(high_gap_rows["n_names"].mean()) if len(high_gap_rows) else np.nan,
        "min_high_gap_n_names": int(high_gap_rows["n_names"].min()) if len(high_gap_rows) else 0,
        "zero_high_gap_name_days": int(high_gap_rows["n_names"].eq(0).sum()) if len(high_gap_rows) else 0,
    }


def build_index_daily(date_infos: list[dict], args: Namespace) -> pd.DataFrame:
    index = pd.Index([info["trade_date"] for info in date_infos], name="trade_date")
    return simulate_leg_sequence(
        date_infos,
        pd.Series("index", index=index),
        variant="index",
        display_name="指数",
        cost_rate=args.cost_rate,
        charge_cost=False,
    )


def case_cache_key(case: CaseSpec) -> tuple[object, ...]:
    return (
        normalize_code(case.index_code),
        case.start_date,
        case.end_date,
        case.high_quantile,
        case.stock_risk_measure,
        case.gap_definition,
        case.hard_gap_threshold,
        case.max_signal_age_days,
        case.min_fresh_names,
    )


def run_case(
    case: CaseSpec,
    run_args: argparse.Namespace,
    cache: dict[tuple[object, ...], tuple[pd.DataFrame, list[dict]]],
) -> tuple[dict[str, object], pd.DataFrame]:
    args = args_for_case(case, run_args)
    if not args.hmm_path.exists():
        raise FileNotFoundError(f"Missing HMM file for {case.hmm_name}: {args.hmm_path}")

    state = read_market_state(args.hmm_path, args.start_date, args.end_date)
    cache_key = case_cache_key(case)
    if cache_key not in cache:
        panel = build_raw_stock_panel(strategy_args(args), state[["trade_date", "market_regime"]].copy())
        cache[cache_key] = (panel, build_date_infos(panel))
    panel, date_infos = cache[cache_key]
    if not date_infos:
        raise ValueError(f"No trading rows for case {case.case_id}")

    index_daily = build_index_daily(date_infos, args)
    if (
        case.risk_off_leg == "cash"
        and case.gap_refresh == "daily"
        and case.index_code == INDEX_CODE
        and case.hmm_name == "locked_expanding_4state"
    ):
        daily_all = build_core_lines_equivalent(date_infos, state, args)
        core_daily = daily_all.loc[daily_all["variant"].eq("gap_state_core")].copy()
    else:
        core_daily = simulate_core_variant(date_infos, state, args, case.risk_off_leg, case.gap_refresh)

    row: dict[str, object] = {
        "case_id": case.case_id,
        "family": case.family,
        "label": case.label,
        "index_code": normalize_code(case.index_code),
        "index_name": INDEX_NAMES.get(normalize_code(case.index_code), ""),
        "start_date": case.start_date,
        "end_date": case.end_date,
        "cost_rate": case.cost_rate,
        "high_quantile": case.high_quantile,
        "max_signal_age_days": case.max_signal_age_days,
        "min_hold_days": case.min_hold_days,
        "hmm_name": case.hmm_name,
        "risk_off_leg": case.risk_off_leg,
        "gap_refresh": case.gap_refresh,
        "stock_risk_measure": case.stock_risk_measure,
        "gap_definition": case.gap_definition,
        "hard_gap_threshold": (
            case.hard_gap_threshold
            if case.stock_risk_measure == "leverage_gap" and case.gap_definition.startswith("hard_")
            else np.nan
        ),
    }
    row.update(summarize_core(core_daily, index_daily))

    out_daily = core_daily.copy()
    for key, value in row.items():
        if key not in out_daily.columns:
            out_daily[key] = value
    return row, out_daily


def build_core_lines_equivalent(date_infos: list[dict], state: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    from run_core_cs500_gap_state_strategy import build_core_lines

    return build_core_lines(date_infos, state, args)


def add_base_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    base = out.loc[out["case_id"].eq("base")]
    if base.empty:
        return out
    base_row = base.iloc[0]
    for col in ["cum_return_net", "excess_vs_index", "active_ir", "max_drawdown", "ann_turnover", "sum_cost"]:
        out[f"delta_{col}_vs_base"] = pd.to_numeric(out[col], errors="coerce") - float(base_row[col])
    return out


def fmt_pct(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2%}"


def fmt_num(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2f}"


def write_report(summary: pd.DataFrame, output_dir: Path) -> Path:
    view_cols = [
        "case_id",
        "family",
        "label",
        "stock_risk_measure",
        "gap_definition",
        "hard_gap_threshold",
        "cum_return_net",
        "excess_vs_index",
        "active_ir",
        "max_drawdown",
        "ann_turnover",
        "sum_cost",
        "delta_cum_return_net_vs_base",
        "delta_active_ir_vs_base",
        "cash_days",
        "no_high_days",
        "high_gap_days",
        "avg_high_gap_n_names",
        "min_high_gap_n_names",
        "zero_high_gap_name_days",
    ]
    view = summary[view_cols].copy()
    for col in [
        "cum_return_net",
        "excess_vs_index",
        "hard_gap_threshold",
        "max_drawdown",
        "ann_turnover",
        "sum_cost",
        "delta_cum_return_net_vs_base",
    ]:
        view[col] = view[col].map(fmt_pct)
    for col in ["active_ir", "delta_active_ir_vs_base", "avg_high_gap_n_names"]:
        view[col] = view[col].map(fmt_num)

    lines = [
        "# Core CS500 Gap x Market State Robustness",
        "",
        "## Scope",
        "",
        "- Baseline is the locked `gap_state_core`: risk-on holds high-Gap names, risk-off holds cash.",
        "- The checks move one axis at a time from the locked baseline.",
        "- `d*`, `phi0`, and HMM fitting are not recalculated. HMM version checks only swap existing forward-filtered output files.",
        "",
        "## Outputs",
        "",
        "- `core_robustness_summary.csv`: one row per robustness case.",
        "- `core_robustness_daily.csv`: daily net-return rows for the core leg in every case.",
        "- `core_robustness_meta.json`: run metadata and HMM file map.",
        "",
        "## Summary",
        "",
        view.to_markdown(index=False),
        "",
    ]
    path = output_dir / "core_robustness_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def main() -> None:
    run_args = parse_args()
    run_args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(run_args.families or [])
    cases = build_cases()
    if selected:
        cases = [case for case in cases if case.family in selected]
    if not cases:
        raise ValueError("No robustness cases selected.")

    rows = []
    daily_parts = []
    cache: dict[tuple[object, ...], tuple[pd.DataFrame, list[dict]]] = {}
    failures = []
    for case in cases:
        try:
            row, daily = run_case(case, run_args, cache)
            rows.append(row)
            daily_parts.append(daily)
            print(f"OK {case.case_id}")
        except Exception as exc:  # Keep long sweeps usable; failures are recorded in metadata.
            failures.append({"case_id": case.case_id, "error": repr(exc)})
            print(f"FAIL {case.case_id}: {exc}")

    if not rows:
        raise RuntimeError(f"All robustness cases failed: {failures}")

    summary = add_base_deltas(pd.DataFrame(rows))
    summary.to_csv(run_args.output_dir / "core_robustness_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_parts, ignore_index=True).to_csv(
        run_args.output_dir / "core_robustness_daily.csv", index=False, encoding="utf-8-sig"
    )
    report_path = write_report(summary, run_args.output_dir)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "output_dir": str(run_args.output_dir),
        "n_cases_requested": len(cases),
        "n_cases_completed": len(rows),
        "failures": failures,
        "hmm_variants": {name: str(path) for name, path in HMM_VARIANTS.items()},
        "notes": [
            "No dstar recalculation.",
            "No phi0 re-estimation.",
            "No HMM retraining.",
            "No smoothed probability inputs.",
        ],
    }
    (run_args.output_dir / "core_robustness_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
