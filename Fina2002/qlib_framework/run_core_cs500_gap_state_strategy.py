from __future__ import annotations

import argparse
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_hs300_gap_state_recognizer import PROJECT_ROOT, setup_matplotlib
from control_hs300_gap_state_turnover import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_COMPONENT_PATH,
    DEFAULT_GAP_PATH,
    DEFAULT_MARKET_DAILY_ROOT,
    DEFAULT_REPT_PATH,
    build_raw_stock_panel,
)


DEFAULT_HMM_PATH = (
    PROJECT_ROOT
    / "market_state_model"
    / "output"
    / "feature_tuned_expanding_hmm_2015"
    / "4state"
    / "market_state_probabilities_4state_feature_tuned_expanding_quarterly.csv"
)
LEGACY_DIRECT_HMM_PATH = (
    PROJECT_ROOT
    / "market_state_model"
    / "output"
    / "feature_tuned_direct_hmm"
    / "4state"
    / "market_state_probabilities_4state_feature_tuned.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "qlib_framework" / "output" / "core_cs500_gap_state_strategy"

INDEX_CODE = "000905"
INDEX_NAME = "CS500"
TRADING_DAYS = 252


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked CS500 Gap x Market State core strategy.")
    parser.add_argument("--index-code", default=INDEX_CODE)
    parser.add_argument("--start-date", default="2021-06-02")
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
    parser.add_argument("--hmm-path", type=Path, default=DEFAULT_HMM_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gap-path", type=Path, default=DEFAULT_GAP_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPT_PATH)
    parser.add_argument("--component-path", type=Path, default=DEFAULT_COMPONENT_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--market-daily-root", type=Path, default=DEFAULT_MARKET_DAILY_ROOT)
    return parser.parse_args()


def cumulative_return(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    return float((1.0 + values).prod() - 1.0) if len(values) else np.nan


def annualized_return(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if values.empty:
        return np.nan
    wealth = float((1.0 + values).prod())
    return float(wealth ** (TRADING_DAYS / len(values)) - 1.0)


def annualized_vol(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(active: pd.Series) -> float:
    values = pd.to_numeric(active, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    sd = values.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return np.nan
    return float(values.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(ret: pd.Series) -> float:
    nav = (1.0 + pd.to_numeric(ret, errors="coerce").fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1.0).min())


def win_rate(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").dropna()
    return float((values > 0).mean()) if len(values) else np.nan


def read_market_state(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    usecols = [
        "date",
        "market_regime",
        "p_high_entropy",
        "p_low_bull",
        "p_low_bear",
        "posterior_entropy",
        "entry_score",
        "transition_score",
        "bullish_transition_signal",
        "bearish_transition_signal",
        "data_split",
    ]
    state = pd.read_csv(path, usecols=lambda col: col in usecols)
    if "date" not in state.columns:
        raise ValueError(f"HMM file has no date column: {path}")
    state["trade_date"] = pd.to_datetime(state["date"], errors="coerce")
    state = state.loc[
        state["trade_date"].ge(pd.Timestamp(start_date))
        & state["trade_date"].le(pd.Timestamp(end_date))
    ].copy()
    state = state.dropna(subset=["trade_date", "market_regime"]).sort_values("trade_date")
    if state.empty:
        raise ValueError("No market-state rows in requested backtest window.")
    return state.reset_index(drop=True)


def strategy_args(args: argparse.Namespace) -> Namespace:
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
        gap_definition=getattr(args, "gap_definition", "rank_signed"),
        hard_gap_threshold=getattr(args, "hard_gap_threshold", 0.10),
    )


def build_date_infos(panel: pd.DataFrame) -> list[dict]:
    infos = []
    for date, part in panel.groupby("trade_date", sort=True):
        returns = dict(zip(part["stock_id"].astype(str), part["change_ratio"].astype(float)))
        targets = {"cash": {}}
        for leg, col in {
            "index": "weight_index",
            "high_gap": "weight_high_gap",
            "no_high": "weight_no_high",
        }.items():
            sub = part.loc[part[col].gt(0), ["stock_id", col]]
            targets[leg] = dict(zip(sub["stock_id"].astype(str), sub[col].astype(float)))
        infos.append({"trade_date": pd.Timestamp(date), "returns": returns, "targets": targets})
    return infos


def generate_risk_gate(state: pd.DataFrame, min_hold_days: int) -> pd.Series:
    regimes = state.set_index("trade_date")["market_regime"].astype(str)
    raw = regimes.shift(1).fillna("Stable").isin(["L+", "Stable"])
    current = True
    held = 0
    out = []
    for desired in raw.tolist():
        candidate = bool(desired)
        if candidate != current and held < min_hold_days:
            candidate = current
        if candidate != current:
            current = candidate
            held = 0
        out.append(current)
        held += 1
    return pd.Series(out, index=regimes.index, name="risk_on")


def drift_weights(weights: dict[str, float], stock_returns: dict[str, float]) -> tuple[dict[str, float], float]:
    if not weights:
        return {}, 0.0
    gross_ret = float(sum(w * stock_returns.get(stock_id, 0.0) for stock_id, w in weights.items()))
    denom = 1.0 + gross_ret
    if denom <= 0:
        return weights.copy(), gross_ret
    drifted = {
        stock_id: float(w * (1.0 + stock_returns.get(stock_id, 0.0)) / denom)
        for stock_id, w in weights.items()
    }
    return {key: value for key, value in drifted.items() if abs(value) > 1e-12}, gross_ret


def simulate_leg_sequence(
    date_infos: list[dict],
    legs: pd.Series,
    variant: str,
    display_name: str,
    cost_rate: float,
    charge_cost: bool = True,
) -> pd.DataFrame:
    previous: dict[str, float] = {}
    rows = []
    leg_map = {pd.Timestamp(k): str(v) for k, v in legs.items()}
    for info in date_infos:
        date = info["trade_date"]
        leg = leg_map.get(date, "index")
        target = info["targets"].get(leg, info["targets"]["index"])
        keys = set(previous) | set(target)
        turnover = float(sum(abs(target.get(key, 0.0) - previous.get(key, 0.0)) for key in keys))
        _, gross_ret = drift_weights(target, info["returns"])
        cost = cost_rate * turnover if charge_cost else 0.0
        net_ret = gross_ret - cost
        drifted, _ = drift_weights(target, info["returns"])
        previous = drifted
        rows.append(
            {
                "trade_date": date,
                "variant": variant,
                "display_name": display_name,
                "selected_leg": leg,
                "gross_return": gross_ret,
                "cost": cost,
                "net_return": net_ret,
                "gross_turnover": turnover,
                "n_names": int(len(target)),
            }
        )
    return pd.DataFrame(rows)


def build_core_lines(date_infos: list[dict], state: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    dates = [info["trade_date"] for info in date_infos]
    index = pd.Index(dates, name="trade_date")
    risk_on = generate_risk_gate(state.loc[state["trade_date"].isin(index)].copy(), args.min_hold_days).reindex(index)
    if risk_on.isna().any():
        raise ValueError("Risk gate is missing dates after aligning market state with CS500 trading days.")

    line_legs = {
        "index": pd.Series("index", index=index),
        "high_gap": pd.Series("high_gap", index=index),
        "market_timing": pd.Series(np.where(risk_on, "index", "cash"), index=index),
        "gap_state_core": pd.Series(np.where(risk_on, "high_gap", "cash"), index=index),
    }
    names = {
        "index": "Index",
        "high_gap": "High Gap",
        "market_timing": "HMM Timing",
        "gap_state_core": "HMM + High Gap",
    }
    parts = []
    for variant, legs in line_legs.items():
        parts.append(
            simulate_leg_sequence(
                date_infos,
                legs,
                variant=variant,
                display_name=names[variant],
                cost_rate=args.cost_rate,
                charge_cost=variant != "index",
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out = out.merge(
        risk_on.rename("risk_on").reset_index(),
        on="trade_date",
        how="left",
    )
    return out


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    index_ret = daily.loc[daily["variant"].eq("index"), ["trade_date", "net_return"]].rename(
        columns={"net_return": "index_net_return"}
    )
    rows = []
    for variant, part in daily.groupby("variant", sort=False):
        merged = part.merge(index_ret, on="trade_date", how="left")
        active = merged["net_return"] - merged["index_net_return"]
        rows.append(
            {
                "variant": variant,
                "display_name": part["display_name"].iloc[0],
                "n_days": int(len(part)),
                "cum_return_net": cumulative_return(part["net_return"]),
                "cum_return_gross": cumulative_return(part["gross_return"]),
                "excess_vs_index": cumulative_return(part["net_return"]) - cumulative_return(merged["index_net_return"]),
                "ann_return": annualized_return(part["net_return"]),
                "ann_vol": annualized_vol(part["net_return"]),
                "active_ir": information_ratio(active),
                "max_drawdown": max_drawdown(part["net_return"]),
                "win_rate": win_rate(part["net_return"]),
                "avg_turnover": float(part["gross_turnover"].mean()),
                "ann_turnover": float(part["gross_turnover"].mean() * TRADING_DAYS),
                "sum_cost": float(part["cost"].sum()),
                "cash_days": int(part["selected_leg"].eq("cash").sum()),
                "high_gap_days": int(part["selected_leg"].eq("high_gap").sum()),
                "index_days": int(part["selected_leg"].eq("index").sum()),
            }
        )
    summary = pd.DataFrame(rows)
    wide = daily.pivot_table(index="trade_date", columns="variant", values="net_return", aggfunc="first").dropna()
    nav = (1.0 + wide).cumprod()
    top = nav.idxmax(axis=1)
    top_share = top.value_counts(normalize=True).rename("top_nav_share").reset_index()
    top_share.columns = ["variant", "top_nav_share"]
    return summary.merge(top_share, on="variant", how="left").fillna({"top_nav_share": 0.0})


def add_curves(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.sort_values(["variant", "trade_date"]).copy()
    out["nav_net"] = out.groupby("variant")["net_return"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    index_nav = out.loc[out["variant"].eq("index"), ["trade_date", "nav_net"]].rename(
        columns={"nav_net": "index_nav"}
    )
    out = out.merge(index_nav, on="trade_date", how="left")
    out["active_nav_vs_index"] = out["nav_net"] / out["index_nav"] - 1.0
    return out


def plot_core(curves: pd.DataFrame, output_dir: Path) -> Path:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    order = ["index", "high_gap", "market_timing", "gap_state_core"]
    colors = {
        "index": "#334155",
        "high_gap": "#7c3aed",
        "market_timing": "#dc2626",
        "gap_state_core": "#0f766e",
    }
    labels = {
        "index": "Index",
        "high_gap": "High Gap",
        "market_timing": "HMM Timing",
        "gap_state_core": "HMM + High Gap",
    }
    for variant in order:
        part = curves.loc[curves["variant"].eq(variant)].sort_values("trade_date")
        ax.plot(part["trade_date"], part["nav_net"], label=labels[variant], color=colors[variant], linewidth=2.0)
    ax.set_title("CS500: Index vs High Gap vs HMM Timing vs HMM + High Gap")
    ax.set_ylabel("Net asset value")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output_dir / "core_cs500_gap_state_strategy_nav.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.2%}"


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    plot_path: Path,
    output_dir: Path,
) -> Path:
    view = summary.copy()
    for col in [
        "cum_return_net",
        "cum_return_gross",
        "excess_vs_index",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "win_rate",
        "avg_turnover",
        "sum_cost",
        "top_nav_share",
    ]:
        view[col] = view[col].map(fmt_pct)
    view["active_ir"] = view["active_ir"].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.2f}")

    risk_counts = curves.loc[curves["variant"].eq("gap_state_core"), "selected_leg"].value_counts().to_dict()
    lines = [
        "# Core CS500 Gap x Market State Strategy",
        "",
        "## Locked Version",
        "",
        "- Universe/index: `000905` CS500.",
        f"- Gap source: `{args.gap_path}`.",
        "- Announcement source: `QUANT_REPT_PATH`, using `Annodt` through the PIT builder in `control_hs300_gap_state_turnover.py`.",
        f"- Market state source: `{args.hmm_path}`.",
        "- Market-state signal uses forward-filtered HMM states only. Smoothed probabilities are not used.",
        f"- Backtest window: `{args.start_date}` to `{args.end_date}`.",
        f"- Trading cost for active lines: `{args.cost_rate:.2%}` times gross turnover. The index benchmark line is treated as buy-and-hold benchmark and is not charged strategy trading cost.",
        f"- Stock risk measure: `{getattr(args, 'stock_risk_measure', 'leverage_gap')}`.",
        f"- High-risk bucket: top `{args.high_quantile:.0%}` names by `{getattr(args, 'stock_risk_measure', 'leverage_gap')}` within current CS500 constituents.",
        f"- PIT Gap freshness: max age `{args.max_signal_age_days}` days when `stock_risk_measure=leverage_gap`; market-data proxies use lagged market data only.",
        f"- Gap definition: `{getattr(args, 'gap_definition', 'rank_signed')}`"
        + (
            f", hard threshold `{getattr(args, 'hard_gap_threshold', 0.10):.2%}`."
            if str(getattr(args, "gap_definition", "rank_signed")).startswith("hard_")
            else "."
        ),
        f"- Risk gate: use yesterday's market regime, with `{args.min_hold_days}` trading-day minimum holding constraint. `L+` and `Stable` are risk-on; `H` and `L-` are risk-off.",
        "",
        "## Four Lines",
        "",
        "- Index: buy-and-hold CS500 benchmark.",
        "- High Gap: always hold the CS500 high-Gap basket.",
        "- HMM Timing: risk-on hold CS500, risk-off hold cash.",
        "- HMM + High Gap: risk-on hold the CS500 high-Gap basket, risk-off hold cash.",
        "",
        "## Results",
        "",
        view[
            [
                "variant",
                "display_name",
                "cum_return_net",
                "excess_vs_index",
                "max_drawdown",
                "active_ir",
                "avg_turnover",
                "sum_cost",
                "top_nav_share",
                "cash_days",
                "high_gap_days",
                "index_days",
            ]
        ].to_markdown(index=False),
        "",
        "## Core Decision",
        "",
        "- Locked core strategy: `gap_state_core`.",
        "- Interpretation: when market state is good, enter high-Gap names inside CS500; when market state is bad, hold cash.",
        f"- Core strategy leg counts: `{json.dumps(risk_counts, ensure_ascii=False)}`.",
        "",
        "## Robustness Tests To Run Next",
        "",
        "1. Cost robustness: 0, 0.03%, 0.05%, 0.10%.",
        "2. Gap bucket robustness: top 10%, 20%, 30%.",
        "3. PIT freshness robustness: max signal age 365, 540, full.",
        "4. Risk-gate robustness: min hold 3, 5, 10, 20 trading days.",
        "5. Risk-off leg robustness: cash versus no-high basket.",
        "6. Index universe robustness: HS300, CS500, CSI1000, CSI800, CSI All Share.",
        "7. Locked HMM convention: use the expanding 4-state HMM only; older HMM outputs are archived comparisons.",
        "8. Subperiod robustness: 2021, 2022, 2023, 2024 and drawdown/recovery windows.",
        "9. Rebalance robustness: daily PIT versus monthly and quarterly high-Gap refresh.",
        "",
        "## Figure",
        "",
        f"- `{plot_path}`",
    ]
    path = output_dir / "core_cs500_gap_state_strategy_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = read_market_state(args.hmm_path, args.start_date, args.end_date)
    panel = build_raw_stock_panel(strategy_args(args), state[["trade_date", "market_regime"]].copy())
    date_infos = build_date_infos(panel)
    daily = build_core_lines(date_infos, state, args)
    summary = summarize(daily)
    curves = add_curves(daily)
    plot_path = plot_core(curves, args.output_dir)

    daily.to_parquet(args.output_dir / "core_cs500_gap_state_strategy_daily.parquet", index=False)
    daily.to_csv(args.output_dir / "core_cs500_gap_state_strategy_daily.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(args.output_dir / "core_cs500_gap_state_strategy_curves.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "core_cs500_gap_state_strategy_summary.csv", index=False, encoding="utf-8-sig")
    report_path = write_report(args, summary, curves, plot_path, args.output_dir)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "index_code": args.index_code,
        "locked_core_variant": "gap_state_core",
        "no_dstar_recalculation": True,
        "no_phi0_reestimate": True,
        "no_hmm_retraining": True,
        "no_smoothed_probabilities": True,
        "stock_risk_measure": args.stock_risk_measure,
        "gap_definition": args.gap_definition,
        "hard_gap_threshold": args.hard_gap_threshold,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "core_cs500_gap_state_strategy_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
