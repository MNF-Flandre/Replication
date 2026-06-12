from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rust_accel import rolling_max_drawdown_grouped as rust_rolling_max_drawdown_grouped
    from rust_accel import rust_available
except ImportError:  # pragma: no cover - package-style imports
    try:
        from qlib_framework.rust_accel import rolling_max_drawdown_grouped as rust_rolling_max_drawdown_grouped
        from qlib_framework.rust_accel import rust_available
    except ImportError:
        rust_rolling_max_drawdown_grouped = None

        def rust_available() -> bool:
            return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMAL_MODEL_DIR = PROJECT_ROOT / "optimal_leverage_model"

DEFAULT_SIGNAL_PATH = (
    PROJECT_ROOT
    / "qlib_framework"
    / "output"
    / "hs300_gap_state_recognizer"
    / "hs300_gap_state_recognizer_best_daily.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "qlib_framework" / "output" / "hs300_gap_state_turnover_control"
DEFAULT_GAP_PATH = OPTIMAL_MODEL_DIR / "output" / "variant_A_clean_main_results.csv"
DEFAULT_REPT_PATH = Path(
    os.environ.get("QUANT_REPT_PATH", PROJECT_ROOT / "external_data" / "3tables" / "IAR_Rept.csv")
).expanduser()
DEFAULT_COMPONENT_PATH = (
    PROJECT_ROOT / "etf_weight" / "output" / "resset_index_returns" / "resset_component_intervals.csv"
)
DEFAULT_BENCHMARK_PATH = (
    PROJECT_ROOT / "etf_weight" / "output" / "resset_index_returns" / "resset_index_daily_returns.csv"
)
DEFAULT_MARKET_DAILY_ROOT = Path(
    os.environ.get("QUANT_MARKET_DAILY_ROOT", PROJECT_ROOT / "external_data" / "market_daily")
).expanduser()

TRADING_DAYS = 252
DEFAULT_COST_RATE = 0.0005
LEG_COLUMNS = {
    "index": "weight_index",
    "high_gap": "weight_high_gap",
    "no_high": "weight_no_high",
    "cash": "",
}
PRED_COLUMNS = {
    "index": "pred_index",
    "high_gap": "pred_high_gap",
    "no_high": "pred_no_high",
}


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    exposure: float = 1.0
    turnover_cap: float | None = None
    confirm_days: int = 1
    min_hold_days: int = 1
    margin: float = 0.0
    rebalance_freq: str = "daily"
    direct_cross: str = "allow"  # allow, block, via_index
    static_leg: str | None = None
    hmm_gap_rule: bool = False
    pure_hmm_index: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control turnover for the HS300 Gap state recognizer without retraining the recognizer."
    )
    parser.add_argument("--signal-path", type=Path, default=DEFAULT_SIGNAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-code", default="000300")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--cost-rate", type=float, default=DEFAULT_COST_RATE)
    parser.add_argument("--gap-path", type=Path, default=DEFAULT_GAP_PATH)
    parser.add_argument(
        "--gap-definition",
        choices=["rank_signed", "rank_signed_raw", "rank_abs", "hard_signed", "hard_abs"],
        default="rank_signed",
    )
    parser.add_argument("--hard-gap-threshold", type=float, default=0.10)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPT_PATH)
    parser.add_argument("--component-path", type=Path, default=DEFAULT_COMPONENT_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--market-daily-root", type=Path, default=DEFAULT_MARKET_DAILY_ROOT)
    parser.add_argument("--high-quantile", type=float, default=0.20)
    parser.add_argument(
        "--stock-risk-measure",
        choices=["leverage_gap", "prev_quarter_volatility", "prev_quarter_max_drawdown"],
        default="leverage_gap",
    )
    parser.add_argument("--max-signal-age-days", type=int, default=540)
    parser.add_argument("--min-fresh-names", type=int, default=30)
    return parser.parse_args()


def normalize_code(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text


def cumulative_return(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    return float((1.0 + values).prod() - 1.0)


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
    wealth = (1.0 + pd.to_numeric(ret, errors="coerce").fillna(0.0)).cumprod()
    if wealth.empty:
        return np.nan
    return float((wealth / wealth.cummax() - 1.0).min())


def win_rate(ret: pd.Series) -> float:
    values = pd.to_numeric(ret, errors="coerce").dropna()
    return float((values > 0).mean()) if len(values) else np.nan


def read_benchmark(path: Path, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    usecols = ["index_code", "index_name", "trade_date", "market_cap_weighted_return"]
    df = pd.read_csv(path, usecols=usecols, dtype={"index_code": "string"})
    df["index_code"] = df["index_code"].map(normalize_code)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.loc[
        df["index_code"].eq(normalize_code(index_code))
        & df["trade_date"].ge(pd.Timestamp(start_date))
        & df["trade_date"].le(pd.Timestamp(end_date))
    ].copy()
    if df.empty:
        raise ValueError("No benchmark rows matched the requested index/date filter.")
    return df.sort_values("trade_date").reset_index(drop=True)


def read_components(path: Path, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"index_code": "string", "stock_id": "string"},
        parse_dates=["begin_date", "end_date_filled"],
    )
    df["index_code"] = df["index_code"].map(normalize_code)
    df["stock_id"] = df["stock_id"].map(normalize_code)
    df = df.loc[
        df["index_code"].eq(normalize_code(index_code))
        & df["end_date_filled"].ge(pd.Timestamp(start_date))
        & df["begin_date"].le(pd.Timestamp(end_date))
    ].copy()
    if df.empty:
        raise ValueError("No component rows matched the requested index/date filter.")
    return df


def read_market_daily(
    root: Path,
    stock_ids: set[str],
    start_date: str,
    end_date: str,
    lookback_days: int = 10,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    lookback_start = start - pd.Timedelta(days=lookback_days)
    parts = []
    for year in range(lookback_start.year, end.year + 1):
        year_dir = root / f"year={year}"
        paths = sorted(year_dir.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No market parquet files under {year_dir}")
        for path in paths:
            part = pd.read_parquet(
                path,
                columns=["stock_id", "trade_date", "change_ratio", "daily_market_cap", "trade_status"],
            )
            part["stock_id"] = part["stock_id"].map(normalize_code)
            part = part.loc[part["stock_id"].isin(stock_ids)].copy()
            if part.empty:
                continue
            part["trade_date"] = pd.to_datetime(part["trade_date"], errors="coerce")
            part = part.loc[part["trade_date"].ge(lookback_start) & part["trade_date"].le(end)].copy()
            if not part.empty:
                parts.append(part)
    if not parts:
        raise ValueError("No market daily rows matched index constituents.")
    market = pd.concat(parts, ignore_index=True)
    market["change_ratio"] = pd.to_numeric(market["change_ratio"], errors="coerce")
    market["daily_market_cap"] = pd.to_numeric(market["daily_market_cap"], errors="coerce")
    market = market.sort_values(["stock_id", "trade_date"]).reset_index(drop=True)
    market["lag_daily_market_cap"] = market.groupby("stock_id")["daily_market_cap"].shift(1)
    shifted_return = market.groupby("stock_id")["change_ratio"].shift(1)
    market["prev_quarter_volatility"] = shifted_return.groupby(market["stock_id"]).transform(
        lambda s: s.rolling(63, min_periods=40).std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    market["prev_quarter_max_drawdown"] = compute_prev_quarter_max_drawdown(market, shifted_return)
    market["prev_quarter_vol_obs"] = shifted_return.groupby(market["stock_id"]).transform(
        lambda s: s.rolling(63, min_periods=1).count()
    )
    market = market.loc[market["trade_date"].ge(start)].copy()
    return market


def rolling_max_drawdown_from_returns(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.nan
    wealth = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(-np.min(drawdown))


def compute_prev_quarter_max_drawdown(market: pd.DataFrame, shifted_return: pd.Series) -> pd.Series:
    if rust_rolling_max_drawdown_grouped is not None and rust_available():
        group_ids = pd.factorize(market["stock_id"], sort=False)[0].astype(np.int64, copy=False)
        try:
            values = rust_rolling_max_drawdown_grouped(
                shifted_return.to_numpy(dtype=np.float64, copy=False),
                group_ids,
                window=63,
                min_periods=40,
            )
            return pd.Series(values, index=market.index)
        except Exception:
            pass
    return shifted_return.groupby(market["stock_id"]).transform(
        lambda s: s.rolling(63, min_periods=40).apply(rolling_max_drawdown_from_returns, raw=True)
    )


def read_gap_signals(gap_path: Path, report_path: Path, trading_dates: np.ndarray) -> pd.DataFrame:
    needed = [
        "firm_id",
        "period_date",
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "optimal_debt_ratio_raw",
        "leverage_gap_raw",
        "leverage_gap",
        "leverage_status",
    ]
    gap = pd.read_csv(gap_path, usecols=lambda col: col in needed)
    gap["stock_id"] = gap["firm_id"].map(normalize_code)
    gap["period_date"] = pd.to_datetime(gap["period_date"], errors="coerce")
    for col in [
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "optimal_debt_ratio_raw",
        "leverage_gap_raw",
        "leverage_gap",
    ]:
        if col in gap.columns:
            gap[col] = pd.to_numeric(gap[col], errors="coerce")
    if "optimal_debt_ratio_raw" not in gap.columns:
        gap["optimal_debt_ratio_raw"] = gap.get("optimal_debt_ratio")
    if "leverage_gap" not in gap.columns:
        gap["leverage_gap"] = gap["observed_debt_ratio"] - gap["optimal_debt_ratio"]
    if "leverage_gap_raw" not in gap.columns:
        gap["leverage_gap_raw"] = gap["observed_debt_ratio"] - gap["optimal_debt_ratio_raw"]
    gap = gap.dropna(subset=["stock_id", "period_date", "leverage_gap"]).copy()

    disclosure = pd.read_csv(report_path, usecols=["Stkcd", "Accper", "Annodt"], low_memory=False)
    disclosure["stock_id"] = disclosure["Stkcd"].map(normalize_code)
    disclosure["period_date"] = pd.to_datetime(disclosure["Accper"], errors="coerce")
    disclosure["announcement_date"] = pd.to_datetime(disclosure["Annodt"], errors="coerce")
    disclosure = disclosure.dropna(subset=["stock_id", "period_date", "announcement_date"]).copy()
    disclosure = disclosure.groupby(["stock_id", "period_date"], as_index=False).agg(
        announcement_date=("announcement_date", "min")
    )
    gap = gap.merge(disclosure, on=["stock_id", "period_date"], how="left")
    gap["valid_disclosure"] = gap["announcement_date"].notna() & (gap["announcement_date"] >= gap["period_date"])

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
            "optimal_debt_ratio_raw",
            "leverage_gap",
            "leverage_gap_raw",
            "leverage_status",
        ]
    ]


def attach_signals(panel: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    signal_groups = {
        stock_id: part.drop(columns=["stock_id"]).sort_values("available_trade_date")
        for stock_id, part in signals.groupby("stock_id", sort=False)
    }
    parts = []
    empty_cols = [col for col in signals.columns if col != "stock_id"]
    for stock_id, part in panel.groupby("stock_id", sort=False):
        left = part.sort_values("trade_date").copy()
        left["trade_date"] = pd.to_datetime(left["trade_date"], errors="coerce").astype("datetime64[ns]")
        right = signal_groups.get(stock_id)
        if right is None or right.empty:
            for col in empty_cols:
                left[col] = pd.NaT if "date" in col else np.nan
            parts.append(left)
            continue
        right = right.copy()
        right["available_trade_date"] = pd.to_datetime(
            right["available_trade_date"], errors="coerce"
        ).astype("datetime64[ns]")
        merged = pd.merge_asof(
            left,
            right,
            left_on="trade_date",
            right_on="available_trade_date",
            direction="backward",
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def read_signal(path: Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    needed = [
        "trade_date",
        "index_code",
        "index_name",
        "signal_available",
        "action",
        "market_regime",
        "entry_score_mean20",
        "bullish_transition_past20",
        "bearish_transition_past20",
        "pred_high_gap",
        "pred_no_high",
        "pred_index",
        "index_return",
        "high_gap_return",
        "no_high_return",
        "strategy_net_return",
        "turnover",
        "cost",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in needed, dtype={"index_code": "string"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["index_code"] = df["index_code"].map(normalize_code)
    for col in ["signal_available", "bullish_transition_past20", "bearish_transition_past20"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    for col in ["pred_high_gap", "pred_no_high", "pred_index", "index_return", "high_gap_return", "no_high_return"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if start_date:
        df = df.loc[df["trade_date"].ge(pd.Timestamp(start_date))].copy()
    if end_date:
        df = df.loc[df["trade_date"].le(pd.Timestamp(end_date))].copy()
    if "signal_available" in df.columns:
        first_signal = df.loc[df["signal_available"], "trade_date"].min()
        df = df.loc[df["trade_date"].ge(first_signal)].copy()
    return df.sort_values("trade_date").reset_index(drop=True)


def build_raw_stock_panel(args: argparse.Namespace, daily_signal: pd.DataFrame) -> pd.DataFrame:
    start_date = daily_signal["trade_date"].min().strftime("%Y-%m-%d")
    end_date = daily_signal["trade_date"].max().strftime("%Y-%m-%d")
    benchmark = read_benchmark(args.benchmark_path, args.index_code, start_date, end_date)
    dates = np.sort(benchmark["trade_date"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    components = read_components(args.component_path, args.index_code, start_date, end_date)
    stock_ids = set(components["stock_id"].dropna().unique().tolist())
    stock_risk_measure = getattr(args, "stock_risk_measure", "leverage_gap")
    market_lookback_days = (
        130
        if stock_risk_measure in {"prev_quarter_volatility", "prev_quarter_max_drawdown"}
        else 10
    )
    market = read_market_daily(
        args.market_daily_root,
        stock_ids,
        start_date,
        end_date,
        lookback_days=market_lookback_days,
    )

    panel = market.merge(
        components[["index_code", "index_name", "stock_id", "begin_date", "end_date_filled"]],
        on="stock_id",
        how="inner",
    )
    panel = panel.loc[
        panel["trade_date"].ge(panel["begin_date"]) & panel["trade_date"].le(panel["end_date_filled"])
    ].copy()
    panel = panel.sort_values(["index_code", "trade_date", "stock_id", "begin_date"])
    panel = panel.drop_duplicates(["index_code", "trade_date", "stock_id"], keep="last")
    panel = panel.loc[panel["trade_date"].isin(set(daily_signal["trade_date"]))].copy()
    panel["valid_base_weight"] = (
        panel["lag_daily_market_cap"].gt(0) & panel["change_ratio"].notna() & panel["change_ratio"].gt(-1.0)
    )
    panel["base_weight_raw"] = panel["lag_daily_market_cap"].where(panel["valid_base_weight"])
    denom = panel.groupby("trade_date")["base_weight_raw"].transform("sum")
    panel["base_weight"] = panel["base_weight_raw"] / denom.replace(0.0, np.nan)
    panel["base_weight"] = panel["base_weight"].fillna(0.0)

    panel["stock_risk_measure"] = stock_risk_measure
    gap_definition = getattr(args, "gap_definition", "rank_signed")
    hard_gap_threshold = getattr(args, "hard_gap_threshold", 0.10)

    if stock_risk_measure == "leverage_gap":
        signals = read_gap_signals(args.gap_path, args.report_path, dates)
        panel = attach_signals(panel, signals)
        panel["days_since_gap_available"] = (panel["trade_date"] - panel["available_trade_date"]).dt.days
        if gap_definition in {"rank_signed", "hard_signed"}:
            panel["risk_metric"] = pd.to_numeric(panel["leverage_gap"], errors="coerce")
        elif gap_definition == "rank_signed_raw":
            panel["risk_metric"] = pd.to_numeric(panel["leverage_gap_raw"], errors="coerce")
        elif gap_definition in {"rank_abs", "hard_abs"}:
            panel["risk_metric"] = pd.to_numeric(panel["leverage_gap"], errors="coerce").abs()
        else:
            raise ValueError(f"Unknown gap_definition: {gap_definition}")
        panel["has_risk_signal"] = (
            panel["valid_base_weight"]
            & panel["risk_metric"].notna()
            & panel["days_since_gap_available"].notna()
            & panel["days_since_gap_available"].ge(0)
            & panel["days_since_gap_available"].le(args.max_signal_age_days)
        )
    elif stock_risk_measure == "prev_quarter_volatility":
        panel["available_trade_date"] = pd.NaT
        panel["days_since_gap_available"] = np.nan
        panel["leverage_gap"] = np.nan
        panel["leverage_gap_raw"] = np.nan
        panel["risk_metric"] = pd.to_numeric(panel["prev_quarter_volatility"], errors="coerce")
        panel["has_risk_signal"] = panel["valid_base_weight"] & panel["risk_metric"].notna()
    elif stock_risk_measure == "prev_quarter_max_drawdown":
        panel["available_trade_date"] = pd.NaT
        panel["days_since_gap_available"] = np.nan
        panel["leverage_gap"] = np.nan
        panel["leverage_gap_raw"] = np.nan
        panel["risk_metric"] = pd.to_numeric(panel["prev_quarter_max_drawdown"], errors="coerce")
        panel["has_risk_signal"] = panel["valid_base_weight"] & panel["risk_metric"].notna()
    else:
        raise ValueError(f"Unknown stock_risk_measure: {stock_risk_measure}")

    panel["gap_metric"] = panel["risk_metric"]
    panel["gap_definition"] = gap_definition
    panel["hard_gap_threshold"] = (
        hard_gap_threshold if stock_risk_measure == "leverage_gap" and gap_definition.startswith("hard_") else np.nan
    )
    panel["has_fresh_gap_signal"] = panel["has_risk_signal"]
    panel["fresh_names"] = panel["has_fresh_gap_signal"].groupby(panel["trade_date"]).transform("sum")
    panel["rank_raw"] = np.nan
    fresh = panel["has_fresh_gap_signal"]
    panel.loc[fresh, "rank_raw"] = panel.loc[fresh].groupby("trade_date")["risk_metric"].rank(method="first")
    if stock_risk_measure == "leverage_gap" and gap_definition.startswith("hard_"):
        panel["high_gap_raw"] = (
            panel["has_fresh_gap_signal"]
            & panel["fresh_names"].ge(args.min_fresh_names)
            & panel["risk_metric"].ge(hard_gap_threshold)
        )
    else:
        panel["high_gap_raw"] = (
            panel["has_fresh_gap_signal"]
            & panel["fresh_names"].ge(args.min_fresh_names)
            & ((panel["rank_raw"] / panel["fresh_names"]) >= (1.0 - args.high_quantile))
        )
    panel["high_weight_total"] = panel["base_weight"].where(panel["high_gap_raw"], 0.0).groupby(panel["trade_date"]).transform("sum")
    panel["rest_weight_total"] = panel["base_weight"].where(~panel["high_gap_raw"], 0.0).groupby(panel["trade_date"]).transform("sum")
    panel["weight_index"] = panel["base_weight"]
    panel["weight_high_gap"] = np.where(
        panel["high_gap_raw"] & panel["high_weight_total"].gt(0),
        panel["base_weight"] / panel["high_weight_total"],
        0.0,
    )
    panel["weight_no_high"] = np.where(
        (~panel["high_gap_raw"]) & panel["rest_weight_total"].gt(0),
        panel["base_weight"] / panel["rest_weight_total"],
        0.0,
    )
    return panel.sort_values(["trade_date", "stock_id"]).reset_index(drop=True)


def build_date_infos(panel: pd.DataFrame) -> list[dict]:
    infos = []
    for date, part in panel.groupby("trade_date", sort=True):
        returns = dict(zip(part["stock_id"], part["change_ratio"].astype(float)))
        legs = {}
        for leg, col in LEG_COLUMNS.items():
            if leg == "cash":
                legs[leg] = {}
                continue
            sub = part.loc[part[col].gt(0), ["stock_id", col]]
            legs[leg] = dict(zip(sub["stock_id"], sub[col].astype(float)))
        infos.append({"trade_date": pd.Timestamp(date), "returns": returns, "legs": legs})
    return infos


def blend_weights(index_weights: dict[str, float], target_weights: dict[str, float], exposure: float) -> dict[str, float]:
    if exposure >= 0.999:
        return target_weights.copy()
    keys = set(index_weights) | set(target_weights)
    out = {
        key: (1.0 - exposure) * index_weights.get(key, 0.0) + exposure * target_weights.get(key, 0.0)
        for key in keys
    }
    return {key: value for key, value in out.items() if abs(value) > 1e-12}


def desired_target(info: dict, leg: str, exposure: float) -> dict[str, float]:
    if leg == "cash":
        return {}
    if leg == "index" or exposure >= 0.999:
        return info["legs"][leg].copy()
    return blend_weights(info["legs"]["index"], info["legs"][leg], exposure)


def apply_turnover_cap(
    previous: dict[str, float],
    desired: dict[str, float],
    cap: float | None,
) -> tuple[dict[str, float], float, float]:
    keys = set(previous) | set(desired)
    raw_diff = float(sum(abs(desired.get(k, 0.0) - previous.get(k, 0.0)) for k in keys))
    if cap is None or raw_diff <= cap or raw_diff <= 1e-15:
        return desired.copy(), raw_diff, raw_diff
    alpha = cap / raw_diff
    target = {key: previous.get(key, 0.0) + alpha * (desired.get(key, 0.0) - previous.get(key, 0.0)) for key in keys}
    target = {key: value for key, value in target.items() if abs(value) > 1e-12}
    return target, cap, raw_diff


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
    drifted = {key: value for key, value in drifted.items() if abs(value) > 1e-12}
    return drifted, gross_ret


def variants() -> list[Variant]:
    out = [
        Variant("static_index", "baseline", static_leg="index"),
        Variant("static_no_high", "baseline", static_leg="no_high"),
        Variant("static_high_gap", "baseline", static_leg="high_gap"),
        Variant("hmm_gap_rule", "hmm_baseline", hmm_gap_rule=True),
        Variant("pure_hmm_index_timing", "hmm_baseline", pure_hmm_index=True),
        Variant("hmm_gap_min_hold_5d", "hmm_min_hold", hmm_gap_rule=True, min_hold_days=5),
        Variant("hmm_gap_min_hold_10d", "hmm_min_hold", hmm_gap_rule=True, min_hold_days=10),
        Variant("hmm_gap_min_hold_20d", "hmm_min_hold", hmm_gap_rule=True, min_hold_days=20),
        Variant("pure_hmm_index_min_hold_5d", "hmm_min_hold", pure_hmm_index=True, min_hold_days=5),
        Variant("pure_hmm_index_min_hold_10d", "hmm_min_hold", pure_hmm_index=True, min_hold_days=10),
        Variant("no_limit", "raw"),
        Variant("confirm_2d", "signal_confirm", confirm_days=2),
        Variant("confirm_3d", "signal_confirm", confirm_days=3),
        Variant("confirm_5d", "signal_confirm", confirm_days=5),
        Variant("min_hold_5d", "min_hold", min_hold_days=5),
        Variant("min_hold_10d", "min_hold", min_hold_days=10),
        Variant("min_hold_20d", "min_hold", min_hold_days=20),
        Variant("margin_0p005", "prediction_margin", margin=0.005),
        Variant("margin_0p010", "prediction_margin", margin=0.010),
        Variant("margin_0p020", "prediction_margin", margin=0.020),
        Variant("min_hold_5d_confirm_2d", "combined_signal", min_hold_days=5, confirm_days=2),
        Variant("min_hold_10d_confirm_2d", "combined_signal", min_hold_days=10, confirm_days=2),
        Variant("block_direct_cross", "cross_control", direct_cross="block"),
        Variant("via_index_direct_cross", "cross_control", direct_cross="via_index"),
        Variant("weekly_rebalance", "rebalance_freq", rebalance_freq="weekly"),
        Variant("monthly_rebalance", "rebalance_freq", rebalance_freq="monthly"),
    ]
    for exposure in [0.75, 0.50, 0.25]:
        out.append(Variant(f"exposure_{exposure:.2f}", "exposure", exposure=exposure))
    for cap in [1.00, 0.75, 0.50, 0.25, 0.10]:
        out.append(Variant(f"turnover_cap_{cap:.2f}", "turnover_cap", turnover_cap=cap))
    for exposure in [0.75, 0.50, 0.25]:
        for cap in [1.00, 0.50, 0.25]:
            out.append(Variant(f"exposure_{exposure:.2f}_cap_{cap:.2f}", "exposure_cap", exposure=exposure, turnover_cap=cap))
    out.extend(
        [
            Variant("min_hold_10d_cap_0.50", "combined_execution", min_hold_days=10, turnover_cap=0.50),
            Variant("confirm_2d_cap_0.50", "combined_execution", confirm_days=2, turnover_cap=0.50),
            Variant("min_hold_10d_confirm_2d_cap_0.50", "combined_execution", min_hold_days=10, confirm_days=2, turnover_cap=0.50),
            Variant("exposure_0.50_min_hold_10d_cap_0.50", "combined_execution", exposure=0.50, min_hold_days=10, turnover_cap=0.50),
        ]
    )
    return out


def scheduled_mask(dates: pd.Series, freq: str) -> pd.Series:
    if freq == "daily":
        return pd.Series(True, index=dates.index)
    s = pd.to_datetime(dates)
    if freq == "weekly":
        return s.dt.to_period("W-FRI").ne(s.shift(-1).dt.to_period("W-FRI"))
    if freq == "monthly":
        return s.dt.to_period("M").ne(s.shift(-1).dt.to_period("M"))
    raise ValueError(f"Unknown rebalance frequency: {freq}")


def prediction_edge(row: pd.Series, candidate: str, current: str) -> float:
    cand = pd.to_numeric(row.get(PRED_COLUMNS[candidate]), errors="coerce")
    cur = pd.to_numeric(row.get(PRED_COLUMNS[current]), errors="coerce")
    if pd.isna(cand) or pd.isna(cur):
        return -np.inf
    return float(cand - cur)


def generate_leg_sequence(daily: pd.DataFrame, variant: Variant) -> list[str]:
    if variant.static_leg is not None:
        return [variant.static_leg] * len(daily)
    if variant.hmm_gap_rule:
        regime = daily["market_regime"].shift(1).fillna("Stable").astype(str)
        raw = np.select(
            [regime.eq("L+"), regime.isin(["H", "L-"])],
            ["high_gap", "no_high"],
            default="index",
        ).tolist()
    elif variant.pure_hmm_index:
        regime = daily["market_regime"].shift(1).fillna("Stable").astype(str)
        raw = np.where(regime.isin(["L+", "Stable"]), "index", "cash").tolist()
    else:
        raw = daily["action"].fillna("index").astype(str).tolist()
    sched = scheduled_mask(daily["trade_date"], variant.rebalance_freq).tolist()
    current = "index"
    held = 0
    pending = None
    pending_count = 0
    out = []

    for i, desired in enumerate(raw):
        if desired not in LEG_COLUMNS:
            desired = "index"
        if not sched[i]:
            candidate = current
        else:
            if desired == current:
                pending = None
                pending_count = 0
                candidate = current
            else:
                if desired == pending:
                    pending_count += 1
                else:
                    pending = desired
                    pending_count = 1
                candidate = desired if pending_count >= variant.confirm_days else current

        if candidate != current and held < variant.min_hold_days:
            candidate = current
        if candidate != current and variant.margin > 0:
            edge = prediction_edge(daily.iloc[i], candidate, current)
            if edge < variant.margin:
                candidate = current
        if candidate != current and {candidate, current} == {"high_gap", "no_high"}:
            if variant.direct_cross == "block":
                candidate = current
            elif variant.direct_cross == "via_index":
                candidate = "index"

        if candidate != current:
            current = candidate
            held = 0
            pending = None
            pending_count = 0
        out.append(current)
        held += 1
    return out


def active_weight_distance(weights: dict[str, float], index_weights: dict[str, float]) -> float:
    keys = set(weights) | set(index_weights)
    return 0.5 * float(sum(abs(weights.get(k, 0.0) - index_weights.get(k, 0.0)) for k in keys))


def simulate_variant(
    daily: pd.DataFrame,
    date_infos: list[dict],
    variant: Variant,
    cost_rate: float,
) -> pd.DataFrame:
    legs = generate_leg_sequence(daily, variant)
    previous = date_infos[0]["legs"]["index"].copy()
    rows = []
    for row, info, leg in zip(daily.itertuples(index=False), date_infos, legs):
        desired = desired_target(info, leg, variant.exposure)
        target, turnover, raw_turnover_demand = apply_turnover_cap(previous, desired, variant.turnover_cap)
        drifted, gross_ret = drift_weights(target, info["returns"])
        cost = cost_rate * turnover
        rows.append(
            {
                "trade_date": info["trade_date"],
                "variant": variant.name,
                "family": variant.family,
                "selected_leg": leg,
                "target_exposure": variant.exposure,
                "gross_return": gross_ret,
                "cost": cost,
                "net_return": gross_ret - cost,
                "gross_turnover": turnover,
                "raw_turnover_demand": raw_turnover_demand,
                "active_weight_distance": active_weight_distance(target, info["legs"]["index"]),
                "n_names": int(len(target)),
            }
        )
        previous = drifted
    return pd.DataFrame(rows)


def summarize_variant(sim: pd.DataFrame, index_net: pd.Series, baseline: dict[str, float] | None = None) -> dict[str, object]:
    ret = sim["net_return"].reset_index(drop=True)
    gross = sim["gross_return"].reset_index(drop=True)
    index_net = index_net.reset_index(drop=True)
    active = ret - index_net
    prev_leg = sim["selected_leg"].shift()
    direct_cross_count = int(
        ((prev_leg.eq("high_gap") & sim["selected_leg"].eq("no_high"))
         | (prev_leg.eq("no_high") & sim["selected_leg"].eq("high_gap"))).sum()
    )
    row = {
        "variant": sim["variant"].iloc[0],
        "family": sim["family"].iloc[0],
        "n_days": int(len(sim)),
        "cum_return_net": cumulative_return(ret),
        "cum_return_gross": cumulative_return(gross),
        "excess_vs_net_index": cumulative_return(ret) - cumulative_return(index_net),
        "ann_return": annualized_return(ret),
        "ann_vol": annualized_vol(ret),
        "active_ir": information_ratio(active),
        "max_drawdown": max_drawdown(ret),
        "win_rate": win_rate(ret),
        "avg_gross_turnover": float(sim["gross_turnover"].mean()),
        "ann_gross_turnover": float(sim["gross_turnover"].mean() * TRADING_DAYS),
        "sum_simple_cost": float(sim["cost"].sum()),
        "avg_active_weight_distance": float(sim["active_weight_distance"].mean()),
        "switch_count": int(max(sim["selected_leg"].ne(sim["selected_leg"].shift()).sum() - 1, 0)),
        "direct_high_nohigh_cross_count": direct_cross_count,
        "high_gap_days": int(sim["selected_leg"].eq("high_gap").sum()),
        "no_high_days": int(sim["selected_leg"].eq("no_high").sum()),
        "index_days": int(sim["selected_leg"].eq("index").sum()),
        "cash_days": int(sim["selected_leg"].eq("cash").sum()),
    }
    if baseline:
        base_turnover = baseline.get("avg_gross_turnover", np.nan)
        base_excess = baseline.get("excess_vs_net_index", np.nan)
        row["turnover_reduction_vs_no_limit"] = (
            1.0 - row["avg_gross_turnover"] / base_turnover if base_turnover and not pd.isna(base_turnover) else np.nan
        )
        row["excess_retention_vs_no_limit"] = (
            row["excess_vs_net_index"] / base_excess if base_excess and not pd.isna(base_excess) else np.nan
        )
    return row


def annual_summary(sim_all: pd.DataFrame, index_variant: str = "static_index") -> pd.DataFrame:
    index = sim_all.loc[sim_all["variant"].eq(index_variant), ["trade_date", "net_return"]].rename(
        columns={"net_return": "index_net_return"}
    )
    rows = []
    for (variant, year), part in sim_all.groupby(["variant", sim_all["trade_date"].dt.year], sort=True):
        merged = part.merge(index, on="trade_date", how="left")
        rows.append(
            {
                "variant": variant,
                "year": int(year),
                "cum_return_net": cumulative_return(merged["net_return"]),
                "index_net_return": cumulative_return(merged["index_net_return"]),
                "excess_vs_net_index": cumulative_return(merged["net_return"]) - cumulative_return(merged["index_net_return"]),
                "active_ir": information_ratio(merged["net_return"] - merged["index_net_return"]),
                "avg_gross_turnover": float(merged["gross_turnover"].mean()),
                "sum_simple_cost": float(merged["cost"].sum()),
                "switch_count": int(max(merged["selected_leg"].ne(merged["selected_leg"].shift()).sum() - 1, 0)),
            }
        )
    return pd.DataFrame(rows)


def validate_leg_returns(daily: pd.DataFrame, date_infos: list[dict]) -> pd.DataFrame:
    rows = []
    for info in date_infos:
        row = {"trade_date": info["trade_date"]}
        for leg in LEG_COLUMNS:
            weights = info["legs"][leg]
            row[f"{leg}_return_from_weights"] = float(
                sum(w * info["returns"].get(stock_id, 0.0) for stock_id, w in weights.items())
            )
        rows.append(row)
    calc = pd.DataFrame(rows)
    check = daily.merge(calc, on="trade_date", how="left")
    out = []
    pairs = {
        "index": ("index_return", "index_return_from_weights"),
        "high_gap": ("high_gap_return", "high_gap_return_from_weights"),
        "no_high": ("no_high_return", "no_high_return_from_weights"),
    }
    for leg, (source_col, calc_col) in pairs.items():
        diff = pd.to_numeric(check[source_col], errors="coerce") - pd.to_numeric(check[calc_col], errors="coerce")
        out.append(
            {
                "leg": leg,
                "source_col": source_col,
                "calc_col": calc_col,
                "mean_abs_diff": float(diff.abs().mean()),
                "max_abs_diff": float(diff.abs().max()),
                "corr": float(check[[source_col, calc_col]].corr().iloc[0, 1]),
            }
        )
    return pd.DataFrame(out)


def build_curves(sim_all: pd.DataFrame) -> pd.DataFrame:
    out = sim_all[["trade_date", "variant", "net_return", "gross_return", "gross_turnover", "selected_leg"]].copy()
    out["nav_net"] = out.groupby("variant")["net_return"].transform(lambda s: (1.0 + s).cumprod())
    out["nav_gross"] = out.groupby("variant")["gross_return"].transform(lambda s: (1.0 + s).cumprod())
    index_nav = out.loc[out["variant"].eq("static_index"), ["trade_date", "nav_net"]].rename(
        columns={"nav_net": "index_nav_net"}
    )
    out = out.merge(index_nav, on="trade_date", how="left")
    out["active_nav_vs_index"] = out["nav_net"] / out["index_nav_net"] - 1.0
    return out


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_active(curves: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Path:
    selected = [
        "no_limit",
        "hmm_gap_rule",
        "pure_hmm_index_timing",
        "static_no_high",
        "static_high_gap",
        "turnover_cap_0.50",
        "exposure_0.50_cap_0.50",
    ]
    balanced = pick_balanced(summary)
    if balanced and balanced not in selected:
        selected.append(balanced)
    part = curves.loc[curves["variant"].isin(selected)].copy()
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in part.groupby("variant", sort=False):
        ax.plot(group["trade_date"], group["active_nav_vs_index"], linewidth=1.8, label=name)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("HS300 Gap state turnover control: active NAV vs net index")
    ax.set_ylabel("active NAV")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "figures" / "hs300_gap_state_turnover_active.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_min_hold5d_baselines(curves: pd.DataFrame, output_dir: Path) -> Path:
    selected = [
        "static_index",
        "static_high_gap",
        "hmm_gap_rule",
        "pure_hmm_index_timing",
        "min_hold_5d",
    ]
    labels = {
        "static_index": "指数",
        "static_high_gap": "纯Gap：始终买高Gap",
        "hmm_gap_rule": "HMM择Gap腿",
        "pure_hmm_index_timing": "纯HMM择时买指数",
        "min_hold_5d": "状态识别器+5日最短持有",
    }
    colors = {
        "static_index": "#111827",
        "static_high_gap": "#ea580c",
        "hmm_gap_rule": "#7c3aed",
        "pure_hmm_index_timing": "#64748b",
        "min_hold_5d": "#dc2626",
    }
    part = curves.loc[curves["variant"].isin(selected)].copy()
    plt = setup_matplotlib()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name in selected:
        group = part.loc[part["variant"].eq(name)].sort_values("trade_date")
        if group.empty:
            continue
        axes[0].plot(group["trade_date"], group["nav_net"], linewidth=2.0, label=labels[name], color=colors[name])
        if name != "static_index":
            axes[1].plot(
                group["trade_date"],
                group["active_nav_vs_index"],
                linewidth=2.0,
                label=labels[name],
                color=colors[name],
            )
    axes[0].set_title("沪深300：5日最短持有 vs 纯Gap / HMM / 指数")
    axes[0].set_ylabel("扣费后净值")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("相对净指数")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    path = output_dir / "figures" / "hs300_min_hold5d_vs_gap_hmm_index.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_turnover_scatter(summary: pd.DataFrame, output_dir: Path) -> Path:
    plt = setup_matplotlib()
    plot_df = summary.loc[~summary["variant"].eq("static_index")].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(plot_df["avg_gross_turnover"], plot_df["excess_vs_net_index"], s=45, alpha=0.75)
    for _, row in plot_df.sort_values("excess_vs_net_index", ascending=False).head(8).iterrows():
        ax.annotate(row["variant"], (row["avg_gross_turnover"], row["excess_vs_net_index"]), fontsize=7, alpha=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Turnover vs net excess")
    ax.set_xlabel("average daily gross turnover")
    ax.set_ylabel("net excess vs net index")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "figures" / "hs300_gap_state_turnover_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_min_hold5d_baseline_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    plot_path: Path,
) -> Path:
    selected = [
        "static_index",
        "static_high_gap",
        "hmm_gap_rule",
        "pure_hmm_index_timing",
        "min_hold_5d",
        "min_hold_10d",
    ]
    labels = {
        "static_index": "指数",
        "static_high_gap": "纯Gap：始终买高Gap",
        "hmm_gap_rule": "HMM择Gap腿",
        "pure_hmm_index_timing": "纯HMM择时买指数",
        "min_hold_5d": "状态识别器+5日最短持有",
        "min_hold_10d": "状态识别器+10日最短持有",
    }
    table_df = summary.loc[summary["variant"].isin(selected)].copy()
    table_df["策略"] = table_df["variant"].map(labels)
    table_df = table_df.set_index("variant").loc[selected].reset_index(drop=False)
    table_cols = [
        "策略",
        "cum_return_net",
        "excess_vs_net_index",
        "active_ir",
        "max_drawdown",
        "avg_gross_turnover",
        "sum_simple_cost",
        "switch_count",
        "direct_high_nohigh_cross_count",
        "high_gap_days",
        "no_high_days",
        "index_days",
        "cash_days",
    ]
    annual_df = annual.loc[annual["variant"].isin(selected)].copy()
    annual_df["策略"] = annual_df["variant"].map(labels)
    annual_df = annual_df.sort_values(["year", "variant"])

    five = summary.loc[summary["variant"].eq("min_hold_5d")].iloc[0]
    ten = summary.loc[summary["variant"].eq("min_hold_10d")].iloc[0]
    lines = [
        "# 5日最短持有版本对比",
        "",
        "## 结论先行",
        "",
        f"- `min_hold_5d` 不是不可以用。它在本次 OOS 里净超额最高：{fmt_pct(five['excess_vs_net_index'])}，高于 `min_hold_10d` 的 {fmt_pct(ten['excess_vs_net_index'])}。",
        f"- 我之前更倾向 `min_hold_10d`，原因是它把平均日换手压到 {fmt_pct(ten['avg_gross_turnover'])}，而 `min_hold_5d` 仍有 {fmt_pct(five['avg_gross_turnover'])}；10d 更稳健、更像交易约束，5d 更像收益最优版本。",
        f"- 如果研究目标是展示发现能力，5d 可以作为主展示；如果目标是可交易保守实现，10d 更合适。报告中应同时列出，不应事后只挑 5d。",
        "",
        "## 对比口径",
        "",
        "- 指数：沪深300成分股权重再平衡并扣交易成本后的净指数。",
        "- 纯Gap：始终持有高 Gap 组合。",
        "- HMM择Gap腿：昨日 `L+` 持有高 Gap，昨日 `H/L-` 持有非高 Gap，`Stable` 持有指数。",
        "- 纯HMM择时买指数：昨日 `L+` 或 `Stable` 持有指数，昨日 `H/L-` 空仓。",
        "- 状态识别器+5日最短持有：使用已有高 Gap 专用状态识别器信号，但每次动作至少持有 5 个交易日。",
        "",
        "## 总体表现",
        "",
        md_table(
            table_df[table_cols],
            [
                "cum_return_net",
                "excess_vs_net_index",
                "max_drawdown",
                "avg_gross_turnover",
                "sum_simple_cost",
            ],
            ["active_ir"],
        ),
        "",
        "## 年度表现",
        "",
        md_table(
            annual_df[
                [
                    "策略",
                    "year",
                    "cum_return_net",
                    "index_net_return",
                    "excess_vs_net_index",
                    "active_ir",
                    "avg_gross_turnover",
                    "sum_simple_cost",
                ]
            ],
            ["cum_return_net", "index_net_return", "excess_vs_net_index", "avg_gross_turnover", "sum_simple_cost"],
            ["active_ir"],
        ),
        "",
        "## 图",
        "",
        f"![5d comparison]({plot_path.as_posix()})",
        "",
    ]
    report_path = args.output_dir / "hs300_min_hold5d_vs_gap_hmm_index_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table_df.to_csv(args.output_dir / "hs300_min_hold5d_vs_gap_hmm_index_summary.csv", index=False, encoding="utf-8-sig")
    annual_df.to_csv(args.output_dir / "hs300_min_hold5d_vs_gap_hmm_index_annual.csv", index=False, encoding="utf-8-sig")
    return report_path


def pick_balanced(summary: pd.DataFrame) -> str | None:
    base = summary.loc[summary["variant"].eq("no_limit")]
    if base.empty:
        return None
    base_turnover = float(base["avg_gross_turnover"].iloc[0])
    candidates = summary.loc[
        (~summary["variant"].isin(["static_index", "static_no_high", "static_high_gap", "no_limit"]))
        & summary["excess_vs_net_index"].gt(0)
        & summary["avg_gross_turnover"].le(base_turnover * 0.50)
    ].copy()
    if candidates.empty:
        candidates = summary.loc[
            (~summary["variant"].isin(["static_index", "static_no_high", "static_high_gap", "no_limit"]))
            & summary["excess_vs_net_index"].gt(0)
        ].copy()
    if candidates.empty:
        return None
    candidates = candidates.sort_values(["active_ir", "excess_vs_net_index"], ascending=False)
    return str(candidates["variant"].iloc[0])


def fmt_pct(x) -> str:
    return "" if pd.isna(x) else f"{float(x):.2%}"


def fmt_float(x) -> str:
    return "" if pd.isna(x) else f"{float(x):.3f}"


def md_table(df: pd.DataFrame, pct_cols: list[str], float_cols: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无可用数据_"
    view = df.head(max_rows).copy() if max_rows else df.copy()
    for col in pct_cols:
        if col in view.columns:
            view[col] = view[col].map(fmt_pct)
    for col in float_cols or []:
        if col in view.columns:
            view[col] = view[col].map(fmt_float)
    view = view.fillna("")
    lines = [
        "| " + " | ".join(map(str, view.columns)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in view.columns) + " |")
    return "\n".join(lines)


def write_report(
    args: argparse.Namespace,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    validation: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    base = summary.loc[summary["variant"].eq("no_limit")].iloc[0]
    balanced_name = pick_balanced(summary)
    balanced = summary.loc[summary["variant"].eq(balanced_name)].iloc[0] if balanced_name else None
    top = summary.sort_values(["excess_vs_net_index", "active_ir"], ascending=False).head(12)
    low_turnover = summary.loc[summary["excess_vs_net_index"].gt(0)].sort_values(
        ["avg_gross_turnover", "excess_vs_net_index"], ascending=[True, False]
    ).head(12)
    selected_names = ["static_index", "static_no_high", "static_high_gap", "hmm_gap_rule", "pure_hmm_index_timing", "no_limit"]
    if balanced_name:
        selected_names.append(balanced_name)
    selected_names.extend([
        "min_hold_5d",
        "min_hold_10d",
        "margin_0p005",
        "turnover_cap_0.50",
        "exposure_0.50_cap_0.50",
        "min_hold_10d_confirm_2d_cap_0.50",
    ])
    selected = summary.loc[summary["variant"].isin(dict.fromkeys(selected_names))].copy()

    lines = [
        "# HS300 Gap State Recognizer Turnover Control",
        "",
        "## 口径",
        "",
        f"- 输入信号：`{args.signal_path}`。",
        "- 模型没有重训；本脚本只改变执行层和信号使用层的换手控制。",
        "- 股票池：沪深300成分股。",
        f"- 高 Gap 定义：成分池内按 PIT `leverage_gap` 排名最高 {args.high_quantile:.0%}。",
        "- 可得日：沿用真实公告日 `Annodt` 后第一个交易日可交易的保守口径。",
        f"- stale 控制：只使用 `days_since_gap_available <= {args.max_signal_age_days}` 的 Gap 信号参与高 Gap 排名。",
        f"- 交易成本：双侧 gross turnover 乘以 {args.cost_rate:.2%}。",
        "- 指数基准也按成分股权重每日再平衡并扣交易成本，因此超额收益是相对净指数。",
        "",
        "## 样本",
        "",
        f"- 起止日期：{daily['trade_date'].min().date()} 至 {daily['trade_date'].max().date()}。",
        f"- 交易日数：{len(daily)}。",
        f"- 原始识别器动作分布：{daily['action'].value_counts(normalize=True).to_dict()}。",
        "",
        "## 组合腿收益校验",
        "",
        md_table(validation, ["mean_abs_diff", "max_abs_diff"], ["corr"]),
        "",
        "## 原始识别器与限换手后的核心对比",
        "",
        md_table(
            selected[
                [
                    "variant",
                    "family",
                    "cum_return_net",
                    "excess_vs_net_index",
                    "active_ir",
                    "max_drawdown",
                    "avg_gross_turnover",
                    "ann_gross_turnover",
                    "sum_simple_cost",
                    "turnover_reduction_vs_no_limit",
                    "excess_retention_vs_no_limit",
                    "switch_count",
                    "direct_high_nohigh_cross_count",
                ]
            ],
            [
                "cum_return_net",
                "excess_vs_net_index",
                "max_drawdown",
                "avg_gross_turnover",
                "sum_simple_cost",
                "turnover_reduction_vs_no_limit",
                "excess_retention_vs_no_limit",
            ],
            ["active_ir", "ann_gross_turnover"],
        ),
        "",
        "## 按净超额排序的前 12 个版本",
        "",
        md_table(
            top[
                [
                    "variant",
                    "family",
                    "cum_return_net",
                    "excess_vs_net_index",
                    "active_ir",
                    "avg_gross_turnover",
                    "sum_simple_cost",
                    "turnover_reduction_vs_no_limit",
                    "switch_count",
                    "direct_high_nohigh_cross_count",
                ]
            ],
            [
                "cum_return_net",
                "excess_vs_net_index",
                "avg_gross_turnover",
                "sum_simple_cost",
                "turnover_reduction_vs_no_limit",
            ],
            ["active_ir"],
        ),
        "",
        "## 正超额且低换手版本",
        "",
        md_table(
            low_turnover[
                [
                    "variant",
                    "family",
                    "cum_return_net",
                    "excess_vs_net_index",
                    "active_ir",
                    "avg_gross_turnover",
                    "sum_simple_cost",
                    "turnover_reduction_vs_no_limit",
                    "switch_count",
                    "direct_high_nohigh_cross_count",
                ]
            ],
            [
                "cum_return_net",
                "excess_vs_net_index",
                "avg_gross_turnover",
                "sum_simple_cost",
                "turnover_reduction_vs_no_limit",
            ],
            ["active_ir"],
        ),
        "",
        "## 年度表现",
        "",
        md_table(
            annual.loc[annual["variant"].isin(selected["variant"])][
                [
                    "variant",
                    "year",
                    "cum_return_net",
                    "index_net_return",
                    "excess_vs_net_index",
                    "active_ir",
                    "avg_gross_turnover",
                    "sum_simple_cost",
                ]
            ],
            ["cum_return_net", "index_net_return", "excess_vs_net_index", "avg_gross_turnover", "sum_simple_cost"],
            ["active_ir"],
        ),
        "",
        "## 结论",
        "",
        f"- 原始识别器 `no_limit`：净收益 {fmt_pct(base['cum_return_net'])}，相对净指数超额 {fmt_pct(base['excess_vs_net_index'])}，平均日换手 {fmt_pct(base['avg_gross_turnover'])}，交易成本累计 {fmt_pct(base['sum_simple_cost'])}。",
    ]
    if balanced is not None:
        lines.extend(
            [
                f"- 平衡候选 `{balanced_name}`：净收益 {fmt_pct(balanced['cum_return_net'])}，相对净指数超额 {fmt_pct(balanced['excess_vs_net_index'])}，平均日换手 {fmt_pct(balanced['avg_gross_turnover'])}，相对原始版本降换手 {fmt_pct(balanced['turnover_reduction_vs_no_limit'])}。",
                "- 它不是新的模型参数，只是执行约束；正式主结果仍应同时报告无约束和限换手版本。",
            ]
        )
    lines.extend(
        [
            "- 如果一个版本只是降低换手但净超额明显下降，说明它主要牺牲了状态识别器捕捉到的高 Gap/非高 Gap 切换窗口。",
            "- 如果 `turnover_cap` 或 `exposure_cap` 版本保留了大部分净超额，说明问题更偏执行冲击；如果 `confirm/min_hold` 更好，说明问题更偏信号抖动。",
            "",
            "## 图表",
            "",
        ]
    )
    for path in plot_paths:
        lines.append(f"- `{path}`")

    report_path = args.output_dir / "hs300_gap_state_turnover_control_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    daily = read_signal(args.signal_path, args.start_date, args.end_date)
    if daily.empty:
        raise ValueError("No signal rows available after filtering.")
    args.start_date = daily["trade_date"].min().strftime("%Y-%m-%d")
    args.end_date = daily["trade_date"].max().strftime("%Y-%m-%d")
    args.index_code = normalize_code(args.index_code)

    panel = build_raw_stock_panel(args, daily)
    date_infos = build_date_infos(panel)
    info_dates = pd.Series([info["trade_date"] for info in date_infos], name="trade_date")
    daily = daily.loc[daily["trade_date"].isin(set(info_dates))].copy().sort_values("trade_date").reset_index(drop=True)
    date_infos = [info for info in date_infos if info["trade_date"] in set(daily["trade_date"])]
    if len(daily) != len(date_infos):
        raise ValueError(f"Signal dates and stock panel dates are not aligned: {len(daily)} vs {len(date_infos)}")

    sim_parts = []
    for variant in variants():
        sim_parts.append(simulate_variant(daily, date_infos, variant, args.cost_rate))
    sim_all = pd.concat(sim_parts, ignore_index=True)
    index_net = sim_all.loc[sim_all["variant"].eq("static_index"), "net_return"].reset_index(drop=True)

    base_summary = summarize_variant(sim_all.loc[sim_all["variant"].eq("no_limit")], index_net)
    summaries = []
    for _, part in sim_all.groupby("variant", sort=False):
        summaries.append(summarize_variant(part, index_net, baseline=base_summary))
    summary = pd.DataFrame(summaries).sort_values(["excess_vs_net_index", "active_ir"], ascending=False).reset_index(drop=True)
    annual = annual_summary(sim_all)
    validation = validate_leg_returns(daily, date_infos)
    curves = build_curves(sim_all)

    sim_all.to_parquet(args.output_dir / "hs300_gap_state_turnover_control_daily.parquet", index=False)
    sim_all.to_csv(args.output_dir / "hs300_gap_state_turnover_control_daily.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(args.output_dir / "hs300_gap_state_turnover_control_curves.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "hs300_gap_state_turnover_control_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(args.output_dir / "hs300_gap_state_turnover_control_annual.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(args.output_dir / "hs300_gap_state_turnover_control_validation.csv", index=False, encoding="utf-8-sig")

    panel_cols = [
        "trade_date",
        "stock_id",
        "change_ratio",
        "weight_index",
        "weight_high_gap",
        "weight_no_high",
        "high_gap_raw",
        "base_weight",
        "stock_risk_measure",
        "risk_metric",
        "leverage_gap",
        "lag_daily_market_cap",
        "prev_quarter_volatility",
        "prev_quarter_max_drawdown",
        "prev_quarter_vol_obs",
        "days_since_gap_available",
    ]
    panel[panel_cols].to_parquet(args.output_dir / "hs300_gap_state_turnover_control_stock_weights.parquet", index=False)

    plot_paths = [
        plot_active(curves, summary, args.output_dir),
        plot_turnover_scatter(summary, args.output_dir),
        plot_min_hold5d_baselines(curves, args.output_dir),
    ]
    focused_report = write_min_hold5d_baseline_report(
        args,
        summary,
        annual,
        plot_paths[-1],
    )
    report_path = write_report(args, daily, summary, annual, validation, plot_paths)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signal_path": str(args.signal_path),
        "index_code": args.index_code,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "cost_rate": args.cost_rate,
        "high_quantile": args.high_quantile,
        "stock_risk_measure": args.stock_risk_measure,
        "max_signal_age_days": args.max_signal_age_days,
        "selection_method": "raw_index",
        "no_model_retraining": True,
        "turnover_definition": "stock-level gross sum abs target minus previous drifted weights",
    }
    (args.output_dir / "hs300_gap_state_turnover_control_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "hs300_gap_state_turnover_control_summary.csv")
    print(report_path)
    print(focused_report)
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
