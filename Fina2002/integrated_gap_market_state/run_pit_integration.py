from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

LEVERAGE_PATH = (
    PROJECT_ROOT
    / "optimal_leverage_model"
    / "output"
    / "variant_A_clean_main_results.csv"
)
HMM_PATH = (
    PROJECT_ROOT
    / "market_state_model"
    / "output"
    / "market_state_probabilities.csv"
)
SMOOTHED_DIAGNOSTIC_PATH = (
    PROJECT_ROOT
    / "market_state_model"
    / "output"
    / "smoothed_probabilities_diagnostic_only.csv"
)
IAR_REPT_PATH = Path(
    os.environ.get("QUANT_REPT_PATH", PROJECT_ROOT / "external_data" / "3tables" / "IAR_Rept.csv")
).expanduser()

ANNOUNCEMENT_CANDIDATES = (
    "Annodt",
    "AnnounceDate",
    "AnnouncementDate",
    "DisclosureDate",
    "InfoPublDate",
    "InfoPubDt",
    "PublishDate",
    "DeclareDate",
    "ReportDate",
    "公告日期",
    "披露日期",
    "发布日期",
    "信息发布日期",
)

LEVERAGE_COLUMNS = [
    "firm_id",
    "period_date",
    "period_type",
    "observed_debt_ratio",
    "optimal_debt_ratio",
    "leverage_gap",
    "leverage_status",
    "tax_rate",
    "debt_cost",
    "data_quality_flags",
]

HMM_COLUMNS = [
    "date",
    "p_high_entropy",
    "p_low_bull",
    "p_low_bear",
    "posterior_entropy",
    "entry_score",
    "transition_score",
    "market_regime",
    "bullish_transition_signal",
    "bearish_transition_signal",
]

FUNDAMENTAL_COLUMNS = [
    "firm_id",
    "trade_date",
    "latest_period_date",
    "available_date",
    "days_since_available",
    "period_type",
    "report_kind",
    "observed_debt_ratio",
    "optimal_debt_ratio",
    "leverage_gap",
    "leverage_status",
    "tax_rate",
    "debt_cost",
    "availability_source",
    "available_date_source",
    "available_date_rule",
    "data_quality_flags",
]

DAILY_OUTPUT_COLUMNS = [
    "firm_id",
    "trade_date",
    "latest_period_date",
    "available_date",
    "days_since_available",
    "period_type",
    "observed_debt_ratio",
    "optimal_debt_ratio",
    "leverage_gap",
    "leverage_status",
    "p_high_entropy",
    "p_low_bull",
    "p_low_bear",
    "posterior_entropy",
    "entry_score",
    "entry_score_mean5",
    "entry_score_mean20",
    "p_low_bull_mean5",
    "p_low_bull_mean20",
    "p_high_entropy_mean5",
    "p_high_entropy_mean20",
    "p_low_bear_mean5",
    "p_low_bear_mean20",
    "posterior_entropy_mean20",
    "transition_score",
    "market_regime",
    "market_regime_mode20",
    "bullish_transition_signal",
    "bullish_transition_past5",
    "bullish_transition_past20",
    "bearish_transition_signal",
    "bearish_transition_past5",
    "bearish_transition_past20",
    "days_since_last_bullish_transition",
    "days_since_last_bearish_transition",
    "gap_x_entry_score",
    "gap_x_entry_score_mean20",
    "gap_x_p_low_bull",
    "gap_x_p_low_bull_mean20",
    "gap_x_p_high_entropy",
    "gap_x_p_high_entropy_mean20",
    "gap_x_bullish_transition",
    "gap_x_bullish_transition_past20",
    "gap_x_H",
    "gap_x_Lplus",
    "gap_x_Lminus",
    "availability_source",
    "available_date_source",
    "available_date_rule",
    "report_kind",
    "tax_rate",
    "debt_cost",
    "data_quality_flags",
]

REPORT_KIND_ORDER = {
    "Unknown": 0,
    "Q1": 1,
    "Q3": 2,
    "H1": 3,
    "Annual": 4,
}


@dataclass(frozen=True)
class PitDiagnostics:
    leverage_rows: int
    event_rows_after_dedup: int
    deduplicated_rows_removed: int
    daily_rows: int
    monthly_rows: int
    quarterly_rows: int
    firm_count: int
    trade_day_count: int
    leak_trade_before_available: int
    leak_state_after_trade: int
    smoothed_used: bool
    actual_announcement_matched_rows: int
    actual_announcement_usable_rows: int
    fallback_rows: int
    missing_available_rows: int
    q1_shifted_after_prev_annual_rows: int


def require_columns(columns: Iterable[str], required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda x: bool(x)
        if isinstance(x, (bool, np.bool_))
        else str(x).strip().lower() in {"true", "1", "yes", "y"}
    ).astype(bool)


def next_trading_day(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    calendar_values: np.ndarray,
) -> pd.Series:
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce").astype("datetime64[ns]")
    arr = parsed.to_numpy(dtype="datetime64[ns]")
    out = np.full(len(arr), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid_date = ~pd.isna(parsed).to_numpy()
    idx = np.searchsorted(calendar_values, arr, side="left")
    valid = valid_date & (idx < len(calendar_values))
    out[valid] = calendar_values[idx[valid]]
    return pd.Series(out, index=parsed.index)


def report_kind_from_period(period_date: pd.Series, period_type: pd.Series) -> pd.Series:
    dates = pd.to_datetime(period_date, errors="coerce")
    pt = period_type.astype(str).str.lower().str.strip()
    kind = pd.Series("Unknown", index=period_date.index, dtype="object")
    kind.loc[dates.dt.month.eq(3)] = "Q1"
    kind.loc[dates.dt.month.eq(6)] = "H1"
    kind.loc[dates.dt.month.eq(9)] = "Q3"
    kind.loc[dates.dt.month.eq(12)] = "Annual"
    kind.loc[pt.isin(["annual", "yearly", "annual_report"])] = "Annual"
    return kind


def load_hmm() -> pd.DataFrame:
    if HMM_PATH.name == "smoothed_probabilities_diagnostic_only.csv":
        raise RuntimeError("Do not use smoothed HMM probabilities for formal PIT panel.")
    header = pd.read_csv(HMM_PATH, nrows=0).columns.tolist()
    require_columns(header, HMM_COLUMNS, "HMM market state probabilities")
    hmm = pd.read_csv(HMM_PATH, usecols=HMM_COLUMNS)
    hmm["trade_date"] = pd.to_datetime(hmm.pop("date"), errors="coerce").astype(
        "datetime64[ns]"
    )
    if hmm["trade_date"].isna().any():
        raise ValueError("Some HMM dates could not be parsed.")
    hmm = hmm.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    for col in [
        "p_high_entropy",
        "p_low_bull",
        "p_low_bear",
        "posterior_entropy",
        "entry_score",
        "transition_score",
    ]:
        hmm[col] = pd.to_numeric(hmm[col], errors="coerce")
    hmm["bullish_transition_signal"] = normalize_bool(hmm["bullish_transition_signal"])
    hmm["bearish_transition_signal"] = normalize_bool(hmm["bearish_transition_signal"])
    return hmm.reset_index(drop=True)


def rolling_regime_mode(values: pd.Series, window: int = 20) -> pd.Series:
    result: list[str | float] = []
    arr = values.astype(object).to_numpy()
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        window_values = [v for v in arr[start : i + 1] if pd.notna(v)]
        if not window_values:
            result.append(np.nan)
            continue
        counts: dict[object, int] = {}
        for value in window_values:
            counts[value] = counts.get(value, 0) + 1
        max_count = max(counts.values())
        tied = {key for key, value in counts.items() if value == max_count}
        chosen = next(value for value in reversed(window_values) if value in tied)
        result.append(chosen)
    return pd.Series(result, index=values.index)


def days_since_last_signal(signal: pd.Series) -> pd.Series:
    out = np.full(len(signal), np.nan, dtype="float64")
    last_idx: int | None = None
    for i, value in enumerate(signal.astype(bool).to_numpy()):
        if value:
            last_idx = i
            out[i] = 0
        elif last_idx is not None:
            out[i] = i - last_idx
    return pd.Series(out, index=signal.index)


def add_hmm_window_features(hmm: pd.DataFrame) -> pd.DataFrame:
    result = hmm.copy()
    for col in ["entry_score", "p_low_bull", "p_high_entropy", "p_low_bear"]:
        result[f"{col}_mean5"] = result[col].rolling(5, min_periods=1).mean()
        result[f"{col}_mean20"] = result[col].rolling(20, min_periods=1).mean()
    result["posterior_entropy_mean20"] = (
        result["posterior_entropy"].rolling(20, min_periods=1).mean()
    )
    result["bullish_transition_past5"] = (
        result["bullish_transition_signal"].astype(int).rolling(5, min_periods=1).max()
    ).astype("int8")
    result["bullish_transition_past20"] = (
        result["bullish_transition_signal"].astype(int).rolling(20, min_periods=1).max()
    ).astype("int8")
    result["bearish_transition_past5"] = (
        result["bearish_transition_signal"].astype(int).rolling(5, min_periods=1).max()
    ).astype("int8")
    result["bearish_transition_past20"] = (
        result["bearish_transition_signal"].astype(int).rolling(20, min_periods=1).max()
    ).astype("int8")
    result["days_since_last_bullish_transition"] = days_since_last_signal(
        result["bullish_transition_signal"]
    )
    result["days_since_last_bearish_transition"] = days_since_last_signal(
        result["bearish_transition_signal"]
    )
    result["market_regime_mode20"] = rolling_regime_mode(result["market_regime"], 20)
    result["state_date"] = result["trade_date"]
    return result


def load_leverage() -> tuple[pd.DataFrame, dict[str, object]]:
    header = pd.read_csv(LEVERAGE_PATH, nrows=0).columns.tolist()
    require_columns(header, LEVERAGE_COLUMNS, "A_clean_main leverage result")
    direct_announcement_cols = [col for col in ANNOUNCEMENT_CANDIDATES if col in header]
    usecols = LEVERAGE_COLUMNS + direct_announcement_cols
    leverage = pd.read_csv(LEVERAGE_PATH, usecols=usecols)
    leverage["firm_id"] = pd.to_numeric(leverage["firm_id"], errors="coerce").astype(
        "Int64"
    )
    leverage["period_date"] = pd.to_datetime(
        leverage["period_date"], errors="coerce"
    ).astype("datetime64[ns]")
    if leverage["firm_id"].isna().any() or leverage["period_date"].isna().any():
        raise ValueError("Some leverage firm_id or period_date values are invalid.")
    leverage["firm_id"] = leverage["firm_id"].astype("int32")
    for col in [
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "leverage_gap",
        "tax_rate",
        "debt_cost",
    ]:
        leverage[col] = pd.to_numeric(leverage[col], errors="coerce")
    leverage["report_kind"] = report_kind_from_period(
        leverage["period_date"], leverage["period_type"]
    )
    leverage["period_year"] = leverage["period_date"].dt.year.astype("int16")
    metadata: dict[str, object] = {
        "leverage_direct_announcement_cols": direct_announcement_cols,
    }
    return leverage, metadata


def load_actual_announcements() -> tuple[pd.DataFrame, dict[str, object]]:
    if not IAR_REPT_PATH.exists():
        return pd.DataFrame(), {
            "announcement_source_path": str(IAR_REPT_PATH),
            "announcement_source_found": False,
            "announcement_field": None,
        }
    header = pd.read_csv(IAR_REPT_PATH, nrows=0).columns.tolist()
    if not {"Stkcd", "Accper", "Annodt"}.issubset(header):
        return pd.DataFrame(), {
            "announcement_source_path": str(IAR_REPT_PATH),
            "announcement_source_found": False,
            "announcement_field": None,
        }
    ann = pd.read_csv(IAR_REPT_PATH, usecols=["Stkcd", "Accper", "Annodt"])
    ann["firm_id"] = pd.to_numeric(ann["Stkcd"], errors="coerce")
    ann["period_date"] = pd.to_datetime(ann["Accper"], errors="coerce")
    ann["announcement_date_raw"] = pd.to_datetime(ann["Annodt"], errors="coerce")
    ann = ann.dropna(subset=["firm_id", "period_date", "announcement_date_raw"])
    ann["firm_id"] = ann["firm_id"].astype("int32")
    grouped = (
        ann.groupby(["firm_id", "period_date"], as_index=False)
        .agg(
            announcement_date_raw=("announcement_date_raw", "min"),
            announcement_date_max=("announcement_date_raw", "max"),
            announcement_date_n=("announcement_date_raw", "nunique"),
        )
        .astype({"firm_id": "int32"})
    )
    return grouped, {
        "announcement_source_path": str(IAR_REPT_PATH),
        "announcement_source_found": True,
        "announcement_field": "Annodt",
        "announcement_rows": len(ann),
        "announcement_unique_firm_period_rows": len(grouped),
    }


def regulatory_deadline_calendar_date(events: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    year = events["period_year"].astype(int)
    annual = events["report_kind"].eq("Annual")
    q1 = events["report_kind"].eq("Q1")
    h1 = events["report_kind"].eq("H1")
    q3 = events["report_kind"].eq("Q3")
    out.loc[annual] = pd.to_datetime((year.loc[annual] + 1).astype(str) + "-04-30")
    out.loc[q1] = pd.to_datetime(year.loc[q1].astype(str) + "-04-30")
    out.loc[h1] = pd.to_datetime(year.loc[h1].astype(str) + "-08-31")
    out.loc[q3] = pd.to_datetime(year.loc[q3].astype(str) + "-10-31")
    return out


def attach_available_dates(
    leverage: pd.DataFrame,
    announcements: pd.DataFrame,
    calendar_values: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    events = leverage.copy()
    if not announcements.empty:
        events = events.merge(
            announcements,
            on=["firm_id", "period_date"],
            how="left",
        )
    else:
        events["announcement_date_raw"] = pd.NaT
        events["announcement_date_max"] = pd.NaT
        events["announcement_date_n"] = np.nan

    events["fallback_deadline_calendar_date"] = regulatory_deadline_calendar_date(events)
    events["fallback_available_date"] = next_trading_day(
        events["fallback_deadline_calendar_date"], calendar_values
    ).to_numpy(dtype="datetime64[ns]")
    events["actual_available_date"] = next_trading_day(
        events["announcement_date_raw"], calendar_values
    ).to_numpy(dtype="datetime64[ns]")

    has_actual = events["announcement_date_raw"].notna()
    actual_usable = has_actual & events["actual_available_date"].notna()
    fallback_usable = ~has_actual & events["fallback_available_date"].notna()
    events["available_date"] = pd.NaT
    events.loc[actual_usable, "available_date"] = events.loc[
        actual_usable, "actual_available_date"
    ]
    events.loc[fallback_usable, "available_date"] = events.loc[
        fallback_usable, "fallback_available_date"
    ]

    events["available_date_source"] = "missing"
    events.loc[actual_usable, "available_date_source"] = "actual_announcement_date"
    events.loc[fallback_usable, "available_date_source"] = (
        "regulatory_deadline_fallback"
    )
    events["available_date_rule"] = "missing_no_trading_day_or_unknown_report_kind"
    events.loc[actual_usable, "available_date_rule"] = (
        "IAR_Rept.Annodt_next_trading_day_if_needed"
    )
    fallback_rule_map = {
        "Annual": "annual_next_trading_day_Yplus1_04_30",
        "Q1": "q1_next_trading_day_Y_04_30",
        "H1": "h1_next_trading_day_Y_08_31",
        "Q3": "q3_next_trading_day_Y_10_31",
    }
    for kind, rule in fallback_rule_map.items():
        mask = fallback_usable & events["report_kind"].eq(kind)
        events.loc[mask, "available_date_rule"] = rule

    prev_annual = events.loc[
        events["report_kind"].eq("Annual") & events["available_date"].notna(),
        ["firm_id", "period_year", "available_date"],
    ].copy()
    prev_annual["period_year"] = prev_annual["period_year"] + 1
    prev_annual = prev_annual.rename(
        columns={"available_date": "prev_annual_available_date"}
    )
    events = events.merge(prev_annual, on=["firm_id", "period_year"], how="left")
    q1_shift = (
        events["report_kind"].eq("Q1")
        & events["available_date"].notna()
        & events["prev_annual_available_date"].notna()
        & (events["available_date"] < events["prev_annual_available_date"])
    )
    events.loc[q1_shift, "available_date"] = events.loc[
        q1_shift, "prev_annual_available_date"
    ]
    events.loc[q1_shift, "available_date_rule"] = (
        events.loc[q1_shift, "available_date_rule"]
        + ";max_with_previous_annual_available_date"
    )
    events["availability_source"] = events["available_date_source"]

    counts = {
        "actual_matched_rows": int(has_actual.sum()),
        "actual_usable_rows": int(actual_usable.sum()),
        "fallback_rows": int(fallback_usable.sum()),
        "missing_available_rows": int(events["available_date"].isna().sum()),
        "q1_shifted_after_prev_annual_rows": int(q1_shift.sum()),
    }
    return events, counts


def deduplicate_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = events.loc[events["available_date"].notna()].copy()
    usable["report_priority"] = usable["report_kind"].map(REPORT_KIND_ORDER).fillna(0)
    usable = usable.sort_values(
        [
            "firm_id",
            "available_date",
            "report_priority",
            "period_date",
            "announcement_date_raw",
        ],
        ascending=[True, True, True, True, True],
    )
    duplicate_mask = usable.duplicated(["firm_id", "available_date"], keep="last")
    dropped = usable.loc[duplicate_mask].copy()
    deduped = usable.loc[~duplicate_mask].copy()
    deduped = deduped.sort_values(["firm_id", "available_date", "period_date"])
    return deduped.reset_index(drop=True), dropped.reset_index(drop=True)


def build_daily_fundamental_panel(
    events: pd.DataFrame,
    calendar_values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy().sort_values(["firm_id", "available_date"])
    events["next_available_date"] = events.groupby("firm_id")["available_date"].shift(-1)
    start_idx = np.searchsorted(
        calendar_values,
        events["available_date"].to_numpy(dtype="datetime64[ns]"),
        side="left",
    )
    next_arr = events["next_available_date"].to_numpy(dtype="datetime64[ns]")
    end_idx = np.empty(len(events), dtype=np.int64)
    has_next = ~pd.isna(events["next_available_date"]).to_numpy()
    end_idx[has_next] = np.searchsorted(
        calendar_values, next_arr[has_next], side="left"
    )
    end_idx[~has_next] = len(calendar_values)
    lengths = end_idx - start_idx
    valid = lengths > 0
    events = events.loc[valid].copy()
    start_idx = start_idx[valid]
    end_idx = end_idx[valid]
    lengths = lengths[valid].astype(np.int64)
    events["forward_fill_trading_days"] = lengths.astype("int32")

    total_rows = int(lengths.sum())
    trade_dates = np.empty(total_rows, dtype="datetime64[ns]")
    days_since_available = np.empty(total_rows, dtype=np.int32)
    pos = 0
    for start, end, available_date in zip(
        start_idx,
        end_idx,
        events["available_date"].to_numpy(dtype="datetime64[ns]"),
    ):
        n = int(end - start)
        dates = calendar_values[start:end]
        trade_dates[pos : pos + n] = dates
        days_since_available[pos : pos + n] = (
            (dates - available_date) / np.timedelta64(1, "D")
        ).astype(np.int32)
        pos += n

    repeated: dict[str, np.ndarray] = {
        "firm_id": np.repeat(events["firm_id"].to_numpy(dtype=np.int32), lengths),
        "trade_date": trade_dates,
        "latest_period_date": np.repeat(
            events["period_date"].to_numpy(dtype="datetime64[ns]"), lengths
        ),
        "available_date": np.repeat(
            events["available_date"].to_numpy(dtype="datetime64[ns]"), lengths
        ),
        "days_since_available": days_since_available,
    }
    for col in [
        "period_type",
        "report_kind",
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "leverage_gap",
        "leverage_status",
        "tax_rate",
        "debt_cost",
        "availability_source",
        "available_date_source",
        "available_date_rule",
        "data_quality_flags",
    ]:
        repeated[col] = np.repeat(events[col].to_numpy(), lengths)

    panel = pd.DataFrame(repeated)
    for col in [
        "period_type",
        "report_kind",
        "leverage_status",
        "availability_source",
        "available_date_source",
        "available_date_rule",
        "data_quality_flags",
    ]:
        panel[col] = panel[col].astype("category")
    return panel, events


def add_interactions(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    gap = result["leverage_gap"]
    result["gap_x_entry_score"] = gap * result["entry_score"]
    result["gap_x_entry_score_mean20"] = gap * result["entry_score_mean20"]
    result["gap_x_p_low_bull"] = gap * result["p_low_bull"]
    result["gap_x_p_low_bull_mean20"] = gap * result["p_low_bull_mean20"]
    result["gap_x_p_high_entropy"] = gap * result["p_high_entropy"]
    result["gap_x_p_high_entropy_mean20"] = gap * result["p_high_entropy_mean20"]
    result["gap_x_bullish_transition"] = (
        gap * result["bullish_transition_signal"].astype(float)
    )
    result["gap_x_bullish_transition_past20"] = (
        gap * result["bullish_transition_past20"].astype(float)
    )
    result["gap_x_H"] = gap * result["market_regime"].eq("H").astype(float)
    result["gap_x_Lplus"] = gap * result["market_regime"].eq("L+").astype(float)
    result["gap_x_Lminus"] = gap * result["market_regime"].eq("L-").astype(float)
    return result


def build_rebalance_panels(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = (
        panel[["trade_date"]]
        .drop_duplicates()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    dates["month"] = dates["trade_date"].dt.to_period("M")
    dates["quarter"] = dates["trade_date"].dt.to_period("Q")
    month_end_dates = set(dates.groupby("month")["trade_date"].max())
    quarter_end_dates = set(dates.groupby("quarter")["trade_date"].max())
    monthly = panel.loc[panel["trade_date"].isin(month_end_dates)].copy()
    quarterly = panel.loc[panel["trade_date"].isin(quarter_end_dates)].copy()
    return monthly, quarterly


def expected_report_periods(leverage: pd.DataFrame) -> pd.DataFrame:
    first_last = (
        leverage.groupby("firm_id")["period_year"]
        .agg(first_year="min", last_year="max")
        .reset_index()
    )
    rows = []
    kind_to_month_day = {
        "Q1": (3, 31),
        "H1": (6, 30),
        "Q3": (9, 30),
        "Annual": (12, 31),
    }
    for firm_id, first_year, last_year in first_last.itertuples(index=False):
        for year in range(int(first_year), int(last_year) + 1):
            for kind, (month, day) in kind_to_month_day.items():
                rows.append(
                    {
                        "firm_id": int(firm_id),
                        "period_year": year,
                        "report_kind": kind,
                        "expected_period_date": pd.Timestamp(year, month, day),
                    }
                )
    expected = pd.DataFrame(rows)
    existing = leverage[["firm_id", "period_year", "report_kind"]].drop_duplicates()
    expected = expected.merge(
        existing.assign(report_exists=True),
        on=["firm_id", "period_year", "report_kind"],
        how="left",
    )
    expected["report_exists"] = expected["report_exists"].fillna(False).astype(bool)
    return expected


def missing_report_forward_fill_days(
    expected: pd.DataFrame,
    events: pd.DataFrame,
    calendar_values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = expected.loc[~expected["report_exists"]].copy()
    if missing.empty:
        return missing, pd.DataFrame()
    missing = missing.rename(columns={"expected_period_date": "period_date"})
    missing["period_type"] = missing["report_kind"]
    missing["fallback_deadline_calendar_date"] = regulatory_deadline_calendar_date(
        missing
    )
    missing["expected_available_date"] = next_trading_day(
        missing["fallback_deadline_calendar_date"], calendar_values
    ).to_numpy(dtype="datetime64[ns]")

    real_events = events.loc[
        events["available_date"].notna(), ["firm_id", "available_date"]
    ].copy()
    real_events["available_date"] = pd.to_datetime(
        real_events["available_date"], errors="coerce"
    ).astype("datetime64[ns]")
    real_events = real_events.sort_values(["firm_id", "available_date"])
    rows = []
    for firm_id, part in missing.groupby("firm_id"):
        firm_events = real_events.loc[real_events["firm_id"].eq(firm_id), "available_date"]
        event_dates = pd.to_datetime(firm_events, errors="coerce").to_numpy(
            dtype="datetime64[ns]"
        )
        for item in part.itertuples(index=False):
            raw_expected_available = item.expected_available_date
            if pd.isna(raw_expected_available) or len(event_dates) == 0:
                rows.append(
                    {
                        "firm_id": firm_id,
                        "period_year": item.period_year,
                        "report_kind": item.report_kind,
                        "expected_available_date": raw_expected_available,
                        "missing_forward_fill_trading_days": 0,
                    }
                )
                continue
            expected_available = np.datetime64(raw_expected_available, "ns")
            idx_prev = np.searchsorted(event_dates, expected_available, side="right") - 1
            if idx_prev < 0:
                extra_days = 0
            else:
                idx_next = np.searchsorted(event_dates, expected_available, side="right")
                if idx_next < len(event_dates):
                    end_date = event_dates[idx_next]
                    end_pos = np.searchsorted(calendar_values, end_date, side="left")
                else:
                    end_pos = len(calendar_values)
                start_pos = np.searchsorted(
                    calendar_values, expected_available, side="left"
                )
                extra_days = max(0, int(end_pos - start_pos))
            rows.append(
                {
                    "firm_id": firm_id,
                    "period_year": item.period_year,
                    "report_kind": item.report_kind,
                    "expected_available_date": expected_available,
                    "missing_forward_fill_trading_days": extra_days,
                }
            )
    missing_ffill = pd.DataFrame(rows)
    return missing, missing_ffill


def numeric_distribution(series: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    out: dict[str, float | int] = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }
    return out


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无可用数据。_"
    part = df if max_rows is None else df.head(max_rows)
    return part.to_markdown(index=False, floatfmt=".6f")


def write_reports(
    leverage: pd.DataFrame,
    events_with_dates: pd.DataFrame,
    events_dedup: pd.DataFrame,
    dropped_events: pd.DataFrame,
    interval_events: pd.DataFrame,
    daily_panel: pd.DataFrame,
    monthly_panel: pd.DataFrame,
    quarterly_panel: pd.DataFrame,
    hmm: pd.DataFrame,
    expected_reports: pd.DataFrame,
    missing_ffill: pd.DataFrame,
    diagnostics: PitDiagnostics,
    metadata: dict[str, object],
) -> None:
    source_counts_events = (
        events_with_dates["available_date_source"]
        .value_counts(dropna=False)
        .rename_axis("available_date_source")
        .reset_index(name="financial_event_rows")
    )
    source_counts_daily = (
        daily_panel["available_date_source"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("available_date_source")
        .reset_index(name="daily_panel_rows")
    )
    source_counts = source_counts_events.merge(
        source_counts_daily, on="available_date_source", how="outer"
    ).fillna(0)
    missing_counts = (
        expected_reports.loc[~expected_reports["report_exists"], "report_kind"]
        .value_counts()
        .reindex(["Q1", "H1", "Q3", "Annual"], fill_value=0)
        .rename_axis("report_kind")
        .reset_index(name="missing_report_count")
    )
    missing_count_map = dict(
        zip(missing_counts["report_kind"], missing_counts["missing_report_count"])
    )
    days_dist = pd.DataFrame([numeric_distribution(daily_panel["days_since_available"])])
    firm_avg_days = (
        daily_panel.groupby("firm_id", observed=True)["days_since_available"]
        .mean()
        .reset_index(name="avg_days_since_available")
    )
    firm_avg_dist = pd.DataFrame(
        [numeric_distribution(firm_avg_days["avg_days_since_available"])]
    )
    interval_dist = pd.DataFrame(
        [numeric_distribution(interval_events["forward_fill_trading_days"])]
    )
    missing_ffill_dist = (
        missing_ffill.groupby("report_kind")["missing_forward_fill_trading_days"]
        .apply(lambda s: pd.Series(numeric_distribution(s)))
        .unstack()
        .reset_index()
        if not missing_ffill.empty
        else pd.DataFrame()
    )
    daily_transition_counts = pd.DataFrame(
        [
            {
                "metric": "bullish_transition_signal_nonzero_rows",
                "hmm_days": int(hmm["bullish_transition_signal"].sum()),
                "daily_panel_rows": int(daily_panel["bullish_transition_signal"].sum()),
                "nonzero_gap_interaction_rows": int(
                    (daily_panel["gap_x_bullish_transition"].fillna(0) != 0).sum()
                ),
            },
            {
                "metric": "bullish_transition_past20_nonzero_rows",
                "hmm_days": int(hmm["bullish_transition_past20"].sum()),
                "daily_panel_rows": int(daily_panel["bullish_transition_past20"].sum()),
                "nonzero_gap_interaction_rows": int(
                    (
                        daily_panel["gap_x_bullish_transition_past20"].fillna(0) != 0
                    ).sum()
                ),
            },
            {
                "metric": "bearish_transition_signal_nonzero_rows",
                "hmm_days": int(hmm["bearish_transition_signal"].sum()),
                "daily_panel_rows": int(daily_panel["bearish_transition_signal"].sum()),
                "nonzero_gap_interaction_rows": int(
                    (daily_panel["bearish_transition_signal"].astype(bool)).sum()
                ),
            },
            {
                "metric": "bearish_transition_past20_nonzero_rows",
                "hmm_days": int(hmm["bearish_transition_past20"].sum()),
                "daily_panel_rows": int(daily_panel["bearish_transition_past20"].sum()),
                "nonzero_gap_interaction_rows": int(
                    (daily_panel["bearish_transition_past20"].astype(bool)).sum()
                ),
            },
        ]
    )
    q1_shift_rows = diagnostics.q1_shifted_after_prev_annual_rows
    actual_field_found = bool(metadata.get("announcement_source_found"))
    direct_cols = metadata.get("leverage_direct_announcement_cols", [])
    old_problem_fixed = int(
        (daily_panel["gap_x_bullish_transition_past20"].fillna(0) != 0).sum()
    ) > 0

    pit_alignment = [
        "# PIT 对齐审计报告",
        "",
        "## 公告日来源",
        "",
        f"- A_clean_main 主结果内公告日字段：{direct_cols if direct_cols else '未找到'}",
        f"- 原始财报公告日字段：{'找到 IAR_Rept.Annodt' if actual_field_found else '未找到'}",
        f"- 公告日源文件：`{metadata.get('announcement_source_path')}`",
        f"- 使用平滑概率作为正式变量：{diagnostics.smoothed_used}",
        f"- 平滑概率诊断文件存在：{SMOOTHED_DIAGNOSTIC_PATH.exists()}",
        "",
        "## 可得日规则",
        "",
        "- 真实公告日可匹配时，使用 `IAR_Rept.Annodt`，若公告日不是 HMM 交易日，则推到之后第一个 HMM 交易日。",
        "- 无真实公告日时，使用法规最晚披露日 fallback：年报为次年 04-30，一季报为当年 04-30，半年报为当年 08-31，三季报为当年 10-31；若 deadline 非交易日，推到之后第一个 HMM 交易日。",
        "- 一季报可得日额外约束为不早于上一年度年报可得日。",
        "",
        "## available_date_source 样本数量",
        "",
        markdown_table(source_counts),
        "",
        "## 去重规则",
        "",
        "- 同一企业同一 `available_date` 出现多条财报记录时，按 `Annual > H1 > Q3 > Q1 > Unknown` 的信息优先级保留；同优先级下保留 `period_date` 最新的一条。",
        f"- 去重前可用事件行数：{len(events_with_dates.loc[events_with_dates['available_date'].notna()]):,}",
        f"- 去重后事件行数：{len(events_dedup):,}",
        f"- 被去重移除行数：{len(dropped_events):,}",
        "",
        "## 泄漏检查",
        "",
        f"- `trade_date < available_date` 仍使用该期财报信息的行数：{diagnostics.leak_trade_before_available:,}",
        f"- `state_date > trade_date` 的行数：{diagnostics.leak_state_after_trade:,}",
        f"- Q1 因上一年年报可得日晚于 Q1 可得日而被后移的事件行数：{q1_shift_rows:,}",
        "",
        "## 日频面板规模",
        "",
        f"- 日频面板行数：{diagnostics.daily_rows:,}",
        f"- 企业数量：{diagnostics.firm_count:,}",
        f"- 交易日数量：{diagnostics.trade_day_count:,}",
        f"- 月频 rebalancing 行数：{diagnostics.monthly_rows:,}",
        f"- 季频 rebalancing 行数：{diagnostics.quarterly_rows:,}",
        "",
        "## forward-fill 天数分布",
        "",
        "### 每条日频记录的 days_since_available",
        "",
        markdown_table(days_dist),
        "",
        "### 每个企业平均 days_since_available",
        "",
        markdown_table(firm_avg_dist),
        "",
        "### 每个财报事件 forward-fill 覆盖交易日数",
        "",
        markdown_table(interval_dist),
        "",
        "## 缺失报告统计",
        "",
        "统计口径：对每个企业从首个到最后一个 A_clean_main 报告年份构造应存在的 Q1/H1/Q3/Annual 期别，仅用于缺失审计，不用于补造财报行。",
        "",
        f"- missing_Q1_report_count: {int(missing_count_map.get('Q1', 0)):,}",
        f"- missing_H1_report_count: {int(missing_count_map.get('H1', 0)):,}",
        f"- missing_Q3_report_count: {int(missing_count_map.get('Q3', 0)):,}",
        f"- missing_annual_report_count: {int(missing_count_map.get('Annual', 0)):,}",
        "",
        markdown_table(missing_counts),
        "",
        "### 因缺失报告而继续 forward-fill 的交易日数分布",
        "",
        markdown_table(missing_ffill_dist),
        "",
        "## HMM transition 覆盖",
        "",
        markdown_table(daily_transition_counts),
        "",
        "## 原报告期集成中 gap_x_bullish_transition 为 0 的问题",
        "",
        f"- `gap_x_bullish_transition_past20` 是否产生非零样本：{old_problem_fixed}",
        "- 原报告期集成只在少数财报可得日取单日状态，容易错过稀疏的 transition 日。PIT 日频面板和 `past20` 窗口变量允许 transition 信息在过去窗口内进入研究面板。",
    ]
    (OUTPUT_DIR / "pit_alignment_report.md").write_text(
        "\n".join(pit_alignment) + "\n", encoding="utf-8"
    )

    pit_integration = [
        "# PIT Gap × Market State 集成报告",
        "",
        "## 为什么报告期级别集成不够好",
        "",
        "原报告期级别集成把低频财报点直接对齐到一个市场状态日期，无法表达财报信息在披露后持续可用的 point-in-time 生命周期，也容易在稀疏的 HMM transition 信号上出现错配。正式风险/收益检验需要 `(firm_id, trade_date, features)` 形态的日频研究面板。",
        "",
        "## 真实披露时间如何处理",
        "",
        "`variant_A_clean_main_results.csv` 本身没有公告日字段。本次从原始财报基本情况表 `IAR_Rept.csv` 中使用 `Annodt` 作为真实公告日，并按企业与报告期匹配到 A_clean_main。公告日不是交易日时，特征从之后第一个 HMM 交易日开始生效。",
        "",
        "## 没有真实披露时间时如何 fallback",
        "",
        "无法匹配 `Annodt` 的财报记录使用法规最晚披露日：年报次年 04-30，一季报当年 04-30，半年报当年 08-31，三季报当年 10-31，并统一推到之后第一个 HMM 交易日。一季报还要求不早于上一年年报的可得日。",
        "",
        "## 缺失季度报告如何处理",
        "",
        "脚本不补造任何不存在的 Q1、H1、Q3 或 Annual 记录。某企业某期报告缺失时，日频面板继续使用上一份已经可得的真实财报记录，直到下一份真实存在且可得的报告出现；缺失报告数量和额外 forward-fill 天数已写入 `pit_alignment_report.md`。",
        "",
        "## 低频 d* 和 Gap 如何变成日频 PIT 特征",
        "",
        "每条 A_clean_main 财报记录先得到 `available_date`，再按企业排序。从 `available_date` 起在 HMM 交易日历上 forward-fill 到下一份报告可得日前一交易日。输出中 `latest_period_date` 表示当前交易日使用的最新财报期，`days_since_available` 表示该信息已可得的自然日数。",
        "",
        "## HMM 日频状态如何合并",
        "",
        "HMM 使用正式 forward filtering 输出 `market_state_probabilities.csv`。日频企业特征按 `trade_date` 与 HMM 当日状态精确合并，未使用 `smoothed_probabilities_diagnostic_only.csv`。",
        "",
        "## 交互变量",
        "",
        "已生成 `gap_x_entry_score`、`gap_x_entry_score_mean20`、`gap_x_p_low_bull`、`gap_x_p_low_bull_mean20`、`gap_x_p_high_entropy`、`gap_x_p_high_entropy_mean20`、`gap_x_bullish_transition`、`gap_x_bullish_transition_past20`、`gap_x_H`、`gap_x_Lplus`、`gap_x_Lminus`。",
        "",
        "## 未来信息泄漏",
        "",
        f"- `trade_date < available_date` 行数：{diagnostics.leak_trade_before_available:,}",
        f"- `state_date > trade_date` 行数：{diagnostics.leak_state_after_trade:,}",
        f"- 正式 HMM 输入使用平滑概率：{diagnostics.smoothed_used}",
        "",
        "## 可用于下一步检验的面板",
        "",
        "- 日频 PIT 面板：`daily_pit_gap_market_state_panel.parquet` / `.csv`。",
        "- 月频 rebalancing 面板：`monthly_rebalance_gap_market_state_panel.parquet`。",
        "- 季频 rebalancing 面板：`quarterly_rebalance_gap_market_state_panel.parquet`。",
        "",
        "这些面板可以继续接入企业未来收益、未来波动、未来回撤等标签，进行状态依赖的风险/收益检验。",
    ]
    (OUTPUT_DIR / "pit_integration_report.md").write_text(
        "\n".join(pit_integration) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hmm_raw = load_hmm()
    hmm = add_hmm_window_features(hmm_raw)
    calendar_values = hmm["trade_date"].to_numpy(dtype="datetime64[ns]")

    leverage, leverage_metadata = load_leverage()
    announcements, ann_metadata = load_actual_announcements()
    metadata = {**leverage_metadata, **ann_metadata}

    events_with_dates, availability_counts = attach_available_dates(
        leverage, announcements, calendar_values
    )
    events_dedup, dropped_events = deduplicate_events(events_with_dates)
    daily_fund, interval_events = build_daily_fundamental_panel(
        events_dedup, calendar_values
    )

    daily_panel = daily_fund.merge(
        hmm.drop(columns=[]),
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    daily_panel = add_interactions(daily_panel)
    daily_panel = daily_panel.sort_values(["trade_date", "firm_id"]).reset_index(
        drop=True
    )
    daily_panel = daily_panel[DAILY_OUTPUT_COLUMNS + ["state_date"]]

    monthly_panel, quarterly_panel = build_rebalance_panels(daily_panel)

    expected_reports = expected_report_periods(leverage)
    missing_reports, missing_ffill = missing_report_forward_fill_days(
        expected_reports, events_dedup, calendar_values
    )

    leak_trade_before_available = int(
        (daily_panel["trade_date"] < daily_panel["available_date"]).sum()
    )
    leak_state_after_trade = int((daily_panel["state_date"] > daily_panel["trade_date"]).sum())
    diagnostics = PitDiagnostics(
        leverage_rows=len(leverage),
        event_rows_after_dedup=len(events_dedup),
        deduplicated_rows_removed=len(dropped_events),
        daily_rows=len(daily_panel),
        monthly_rows=len(monthly_panel),
        quarterly_rows=len(quarterly_panel),
        firm_count=int(daily_panel["firm_id"].nunique()),
        trade_day_count=int(daily_panel["trade_date"].nunique()),
        leak_trade_before_available=leak_trade_before_available,
        leak_state_after_trade=leak_state_after_trade,
        smoothed_used=False,
        actual_announcement_matched_rows=availability_counts["actual_matched_rows"],
        actual_announcement_usable_rows=availability_counts["actual_usable_rows"],
        fallback_rows=availability_counts["fallback_rows"],
        missing_available_rows=availability_counts["missing_available_rows"],
        q1_shifted_after_prev_annual_rows=availability_counts[
            "q1_shifted_after_prev_annual_rows"
        ],
    )

    events_with_dates.to_csv(
        OUTPUT_DIR / "pit_financial_event_availability_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_reports.to_csv(
        OUTPUT_DIR / "pit_missing_report_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_ffill.to_csv(
        OUTPUT_DIR / "pit_missing_report_forward_fill_days.csv",
        index=False,
        encoding="utf-8-sig",
    )

    parquet_daily = OUTPUT_DIR / "daily_pit_gap_market_state_panel.parquet"
    csv_daily = OUTPUT_DIR / "daily_pit_gap_market_state_panel.csv"
    monthly_path = OUTPUT_DIR / "monthly_rebalance_gap_market_state_panel.parquet"
    quarterly_path = OUTPUT_DIR / "quarterly_rebalance_gap_market_state_panel.parquet"

    daily_panel.to_parquet(parquet_daily, index=False)
    daily_panel.drop(columns=["state_date"]).to_csv(
        csv_daily, index=False, encoding="utf-8-sig"
    )
    monthly_panel.drop(columns=["state_date"]).to_parquet(monthly_path, index=False)
    quarterly_panel.drop(columns=["state_date"]).to_parquet(quarterly_path, index=False)

    write_reports(
        leverage,
        events_with_dates,
        events_dedup,
        dropped_events,
        interval_events,
        daily_panel,
        monthly_panel,
        quarterly_panel,
        hmm,
        expected_reports,
        missing_ffill,
        diagnostics,
        metadata,
    )

    print("Wrote PIT daily panel:", parquet_daily)
    print("Daily rows:", f"{len(daily_panel):,}")
    print("Monthly rows:", f"{len(monthly_panel):,}")
    print("Quarterly rows:", f"{len(quarterly_panel):,}")
    print("Leak trade_date < available_date:", leak_trade_before_available)
    print("Leak state_date > trade_date:", leak_state_after_trade)
    print("Actual announcement usable rows:", availability_counts["actual_usable_rows"])
    print("Fallback rows:", availability_counts["fallback_rows"])
    print(
        "Bullish transition past20 panel rows:",
        int(daily_panel["bullish_transition_past20"].sum()),
    )


if __name__ == "__main__":
    main()
