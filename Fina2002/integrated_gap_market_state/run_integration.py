from __future__ import annotations

from dataclasses import dataclass
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

ANNOUNCEMENT_DATE_CANDIDATES = (
    "announcement_date",
    "ann_date",
    "announce_date",
    "disclosure_date",
    "report_release_date",
    "publish_date",
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

PANEL_COLUMNS = [
    "firm_id",
    "period_date",
    "available_date",
    "state_date",
    "period_type",
    "observed_debt_ratio",
    "optimal_debt_ratio",
    "leverage_gap",
    "leverage_status",
    "tax_rate",
    "debt_cost",
    "p_high_entropy",
    "p_low_bull",
    "p_low_bear",
    "posterior_entropy",
    "entry_score",
    "transition_score",
    "market_regime",
    "bullish_transition_signal",
    "bearish_transition_signal",
    "is_H",
    "is_Lplus",
    "is_Lminus",
    "gap_x_entry_score",
    "gap_x_p_low_bull",
    "gap_x_p_high_entropy",
    "gap_x_p_low_bear",
    "gap_x_bullish_transition",
    "gap_x_H",
    "gap_x_Lplus",
    "gap_x_Lminus",
    "alignment_method",
    "available_date_source",
    "period_to_state_days",
    "available_to_state_days",
    "state_staleness_flag",
    "data_quality_flags",
]

INTERACTION_COLUMNS = [
    "gap_x_entry_score",
    "gap_x_p_low_bull",
    "gap_x_p_high_entropy",
    "gap_x_p_low_bear",
    "gap_x_bullish_transition",
    "gap_x_H",
    "gap_x_Lplus",
    "gap_x_Lminus",
]

SUMMARY_NUMERIC_COLUMNS = [
    "observed_debt_ratio",
    "optimal_debt_ratio",
    "leverage_gap",
    "tax_rate",
    "debt_cost",
    "p_high_entropy",
    "p_low_bull",
    "p_low_bear",
    "posterior_entropy",
    "entry_score",
    "transition_score",
    *INTERACTION_COLUMNS,
    "period_to_state_days",
    "available_to_state_days",
]


@dataclass(frozen=True)
class AlignmentDiagnostics:
    method: str
    rows: int
    matched_rows: int
    unmatched_rows: int
    unmatched_before_hmm_start: int
    unmatched_missing_alignment_date: int
    future_state_leak_rows: int
    mean_period_to_state_days: float
    mean_available_to_state_days: float
    stale_rows_gt_10d: int
    stale_rows_gt_30d: int
    stale_rows_gt_90d: int


def require_columns(columns: Iterable[str], required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if HMM_PATH.name == "smoothed_probabilities_diagnostic_only.csv":
        raise RuntimeError("Formal integration must not use smoothed HMM probabilities.")

    leverage_header = pd.read_csv(LEVERAGE_PATH, nrows=0).columns.tolist()
    hmm_header = pd.read_csv(HMM_PATH, nrows=0).columns.tolist()
    require_columns(leverage_header, LEVERAGE_COLUMNS, "leverage main result")
    require_columns(hmm_header, HMM_COLUMNS, "HMM market state result")

    announcement_cols = [
        col for col in ANNOUNCEMENT_DATE_CANDIDATES if col in leverage_header
    ]
    read_leverage_cols = LEVERAGE_COLUMNS + announcement_cols

    leverage = pd.read_csv(LEVERAGE_PATH, usecols=read_leverage_cols)
    hmm = pd.read_csv(HMM_PATH, usecols=HMM_COLUMNS)

    leverage["period_date"] = pd.to_datetime(
        leverage["period_date"], errors="coerce"
    ).astype("datetime64[ns]")
    hmm["date"] = pd.to_datetime(hmm["date"], errors="coerce").astype(
        "datetime64[ns]"
    )

    if leverage["period_date"].isna().any():
        raise ValueError("Some leverage period_date values could not be parsed.")
    if hmm["date"].isna().any():
        raise ValueError("Some HMM date values could not be parsed.")

    for col in [
        "observed_debt_ratio",
        "optimal_debt_ratio",
        "leverage_gap",
        "tax_rate",
        "debt_cost",
    ]:
        leverage[col] = pd.to_numeric(leverage[col], errors="coerce")

    for col in [
        "p_high_entropy",
        "p_low_bull",
        "p_low_bear",
        "posterior_entropy",
        "entry_score",
        "transition_score",
    ]:
        hmm[col] = pd.to_numeric(hmm[col], errors="coerce")

    for col in ["bullish_transition_signal", "bearish_transition_signal"]:
        hmm[col] = hmm[col].map(
            lambda x: bool(x)
            if isinstance(x, (bool, np.bool_))
            else str(x).strip().lower() in {"true", "1", "yes", "y"}
        )

    metadata = {
        "leverage_columns": leverage_header,
        "hmm_columns": hmm_header,
        "announcement_cols": announcement_cols,
        "used_smoothed_probabilities": False,
        "smoothed_diagnostic_file_exists": SMOOTHED_DIAGNOSTIC_PATH.exists(),
    }
    return leverage, hmm, metadata


def infer_available_date(leverage: pd.DataFrame, announcement_cols: list[str]) -> pd.DataFrame:
    result = leverage.copy()
    result["available_date_source"] = "rule_no_announcement_date"

    if announcement_cols:
        parsed_candidates = [
            pd.to_datetime(result[col], errors="coerce") for col in announcement_cols
        ]
        announcement_date = parsed_candidates[0]
        for candidate in parsed_candidates[1:]:
            announcement_date = announcement_date.fillna(candidate)
        result["available_date"] = announcement_date
        result.loc[result["available_date"].notna(), "available_date_source"] = (
            "announcement_date"
        )
    else:
        result["available_date"] = pd.NaT

    period_type_normalized = result["period_type"].astype(str).str.lower().str.strip()
    is_annual = (
        period_type_normalized.eq("annual")
        | period_type_normalized.eq("yearly")
        | period_type_normalized.eq("annual_report")
    )
    rule_available_date = result["period_date"] + pd.to_timedelta(
        np.where(is_annual, 120, 45), unit="D"
    )
    result["available_date"] = result["available_date"].fillna(rule_available_date)
    result["available_date"] = pd.to_datetime(
        result["available_date"], errors="coerce"
    ).astype("datetime64[ns]")
    result.loc[
        result["available_date_source"].eq("rule_no_announcement_date"),
        "available_date_source",
    ] = np.where(
        is_annual[result["available_date_source"].eq("rule_no_announcement_date")],
        "rule_period_date_plus_120d",
        "rule_period_date_plus_45d",
    )
    return result


def merge_asof_state(
    leverage: pd.DataFrame,
    hmm: pd.DataFrame,
    alignment_date_col: str,
    alignment_method: str,
) -> pd.DataFrame:
    left = leverage.copy()
    left["_row_id"] = np.arange(len(left))
    left["_alignment_date"] = pd.to_datetime(
        left[alignment_date_col], errors="coerce"
    ).astype("datetime64[ns]")
    right = hmm.rename(columns={"date": "state_date"}).copy()
    right["state_date"] = pd.to_datetime(right["state_date"], errors="coerce").astype(
        "datetime64[ns]"
    )
    right = right.sort_values("state_date")

    merged = pd.merge_asof(
        left.sort_values("_alignment_date"),
        right,
        left_on="_alignment_date",
        right_on="state_date",
        direction="backward",
    )
    merged = merged.sort_values("_row_id").drop(columns=["_row_id"])
    merged["alignment_method"] = alignment_method
    return merged


def add_constructed_variables(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    result["is_H"] = result["market_regime"].eq("H").astype("int8")
    result["is_Lplus"] = result["market_regime"].eq("L+").astype("int8")
    result["is_Lminus"] = result["market_regime"].eq("L-").astype("int8")

    result["gap_x_entry_score"] = result["leverage_gap"] * result["entry_score"]
    result["gap_x_p_low_bull"] = result["leverage_gap"] * result["p_low_bull"]
    result["gap_x_p_high_entropy"] = result["leverage_gap"] * result["p_high_entropy"]
    result["gap_x_p_low_bear"] = result["leverage_gap"] * result["p_low_bear"]
    result["gap_x_bullish_transition"] = (
        result["leverage_gap"] * result["bullish_transition_signal"].astype(float)
    )
    result["gap_x_H"] = result["leverage_gap"] * result["is_H"]
    result["gap_x_Lplus"] = result["leverage_gap"] * result["is_Lplus"]
    result["gap_x_Lminus"] = result["leverage_gap"] * result["is_Lminus"]

    result["period_to_state_days"] = (
        result["state_date"] - result["period_date"]
    ).dt.days
    result["available_to_state_days"] = (
        result["available_date"] - result["state_date"]
    ).dt.days
    result["state_staleness_flag"] = np.select(
        [
            result["state_date"].isna(),
            result["available_to_state_days"] > 90,
            result["available_to_state_days"] > 30,
            result["available_to_state_days"] > 10,
        ],
        ["unmatched", "stale_gt_90d", "stale_gt_30d", "stale_gt_10d"],
        default="ok",
    )
    return result


def diagnostics(
    panel: pd.DataFrame,
    hmm: pd.DataFrame,
    method: str,
    alignment_date_col: str,
) -> AlignmentDiagnostics:
    matched = panel["state_date"].notna()
    alignment_date = panel[alignment_date_col]
    hmm_min = hmm["date"].min()
    future_state_leak = matched & (panel["state_date"] > alignment_date)
    return AlignmentDiagnostics(
        method=method,
        rows=len(panel),
        matched_rows=int(matched.sum()),
        unmatched_rows=int((~matched).sum()),
        unmatched_before_hmm_start=int(((alignment_date < hmm_min) & ~matched).sum()),
        unmatched_missing_alignment_date=int((alignment_date.isna() & ~matched).sum()),
        future_state_leak_rows=int(future_state_leak.sum()),
        mean_period_to_state_days=float(panel.loc[matched, "period_to_state_days"].mean()),
        mean_available_to_state_days=float(
            panel.loc[matched, "available_to_state_days"].mean()
        ),
        stale_rows_gt_10d=int((panel["available_to_state_days"] > 10).sum()),
        stale_rows_gt_30d=int((panel["available_to_state_days"] > 30).sum()),
        stale_rows_gt_90d=int((panel["available_to_state_days"] > 90).sum()),
    )


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            rows.append({"variable": col, "count": 0})
            continue
        row = {
            "variable": col,
            "count": int(s.count()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
        }
        for q, value in s.quantile(quantiles).items():
            row[f"p{int(q * 100):02d}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def gap_by_regime(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, part in panel.groupby("market_regime", dropna=False):
        gap = pd.to_numeric(part["leverage_gap"], errors="coerce")
        status = part["leverage_status"].astype(str)
        rows.append(
            {
                "market_regime": regime,
                "n": len(part),
                "gap_mean": gap.mean(),
                "gap_median": gap.median(),
                "gap_p10": gap.quantile(0.10),
                "gap_p25": gap.quantile(0.25),
                "gap_p75": gap.quantile(0.75),
                "gap_p90": gap.quantile(0.90),
                "over_levered_n": int(status.eq("over_levered").sum()),
                "under_levered_n": int(status.eq("under_levered").sum()),
                "neutral_n": int(status.eq("near_optimal").sum()),
                "over_levered_ratio": status.eq("over_levered").mean(),
                "under_levered_ratio": status.eq("under_levered").mean(),
                "neutral_ratio": status.eq("near_optimal").mean(),
            }
        )
    order = {"H": 0, "L+": 1, "L-": 2}
    out = pd.DataFrame(rows)
    out["_order"] = out["market_regime"].map(order).fillna(99)
    return out.sort_values(["_order", "market_regime"]).drop(columns="_order")


def entry_score_quantile_gap(panel: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    matched = panel.dropna(subset=["entry_score", "leverage_gap"]).copy()
    if matched.empty:
        return pd.DataFrame()
    try:
        matched["entry_score_quantile"] = pd.qcut(
            matched["entry_score"],
            q=bins,
            labels=[f"Q{i}" for i in range(1, bins + 1)],
            duplicates="drop",
        )
    except ValueError:
        matched["entry_score_quantile"] = "all"
    grouped = matched.groupby("entry_score_quantile", observed=False)
    return grouped.agg(
        n=("leverage_gap", "size"),
        entry_score_min=("entry_score", "min"),
        entry_score_max=("entry_score", "max"),
        gap_mean=("leverage_gap", "mean"),
        gap_median=("leverage_gap", "median"),
        gap_p25=("leverage_gap", lambda x: x.quantile(0.25)),
        gap_p75=("leverage_gap", lambda x: x.quantile(0.75)),
    ).reset_index()


def correlation_summary(panel: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("leverage_gap", "entry_score"),
        ("leverage_gap", "p_low_bull"),
        ("leverage_gap", "p_high_entropy"),
        ("leverage_gap", "p_low_bear"),
        ("leverage_gap", "transition_score"),
    ]
    rows = []
    for left, right in pairs:
        pair = panel[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        rows.append(
            {
                "left": left,
                "right": right,
                "n": len(pair),
                "pearson_corr": pair[left].corr(pair[right]) if len(pair) >= 2 else np.nan,
                "abs_corr_gt_0_70": (
                    abs(pair[left].corr(pair[right])) > 0.70 if len(pair) >= 2 else False
                ),
            }
        )
    return pd.DataFrame(rows)


def format_number(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无可用数据。_"
    part = df if max_rows is None else df.head(max_rows)
    return part.to_markdown(index=False, floatfmt=".6f")


def write_alignment_report(
    path: Path,
    leverage: pd.DataFrame,
    hmm: pd.DataFrame,
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    diag_a: AlignmentDiagnostics,
    diag_b: AlignmentDiagnostics,
    gap_regime: pd.DataFrame,
    entry_quantile: pd.DataFrame,
    corr: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    regime_counts = (
        panel_b["market_regime"].value_counts(dropna=False).rename_axis("market_regime")
        .reset_index(name="n")
    )
    bullish_count = int(panel_b["bullish_transition_signal"].fillna(False).sum())
    state_after_available = int((panel_b["state_date"] > panel_b["available_date"]).sum())
    state_after_period_a = int((panel_a["state_date"] > panel_a["period_date"]).sum())
    unknown_period_type = int(
        (
            ~panel_b["period_type"]
            .astype(str)
            .str.lower()
            .str.contains("annual|quarter|semiannual", na=False)
        ).sum()
    )

    unmatched_reasons = pd.DataFrame(
        [
            {
                "alignment_method": diag_a.method,
                "unmatched_rows": diag_a.unmatched_rows,
                "before_hmm_start": diag_a.unmatched_before_hmm_start,
                "missing_alignment_date": diag_a.unmatched_missing_alignment_date,
                "other_or_after_filters": diag_a.unmatched_rows
                - diag_a.unmatched_before_hmm_start
                - diag_a.unmatched_missing_alignment_date,
            },
            {
                "alignment_method": diag_b.method,
                "unmatched_rows": diag_b.unmatched_rows,
                "before_hmm_start": diag_b.unmatched_before_hmm_start,
                "missing_alignment_date": diag_b.unmatched_missing_alignment_date,
                "other_or_after_filters": diag_b.unmatched_rows
                - diag_b.unmatched_before_hmm_start
                - diag_b.unmatched_missing_alignment_date,
            },
        ]
    )

    diag_table = pd.DataFrame([diag_a.__dict__, diag_b.__dict__])
    lines = [
        "# 对齐质量报告",
        "",
        "## 输入样本",
        "",
        f"- 最优负债率主结果：`{LEVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- HMM 正式过滤概率：`{HMM_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 平滑概率诊断文件存在：{metadata['smoothed_diagnostic_file_exists']}；本次读取：False",
        f"- 最优负债率样本行数：{len(leverage):,}",
        f"- HMM 日度状态样本行数：{len(hmm):,}",
        f"- 最优负债率期间范围：{leverage['period_date'].min().date()} 至 {leverage['period_date'].max().date()}",
        f"- HMM 状态日期范围：{hmm['date'].min().date()} 至 {hmm['date'].max().date()}",
        "",
        "## 对齐方式",
        "",
        "- 方式 A：报告期截止日对齐，取 `state_date = max(s: s <= period_date)`。",
        "- 方式 B：报告期后可得日对齐。输入文件未包含公告日期字段，因此使用规则化可得日：季报/半年报 `period_date + 45 days`，年报 `period_date + 120 days`，再取 `state_date = max(s: s <= available_date)`。",
        "- 主输出 `integrated_gap_market_state_panel.csv` 使用方式 B。",
        "",
        "## A/B 对齐诊断",
        "",
        markdown_table(diag_table),
        "",
        "## 未匹配原因",
        "",
        markdown_table(unmatched_reasons),
        "",
        "## 泄漏与时点检查",
        "",
        f"- 方式 A 中 `state_date > period_date` 行数：{state_after_period_a:,}",
        f"- 方式 B 中 `state_date > available_date` 行数：{state_after_available:,}",
        f"- 是否错误使用平滑概率：否。正式输入仅为 `{HMM_PATH.name}`。",
        f"- 方式 B 中 `available_date - state_date > 10` 天行数：{diag_b.stale_rows_gt_10d:,}",
        f"- 方式 B 中 `available_date - state_date > 30` 天行数：{diag_b.stale_rows_gt_30d:,}",
        f"- 方式 B 中 `available_date - state_date > 90` 天行数：{diag_b.stale_rows_gt_90d:,}",
        f"- 未识别 `period_type` 行数：{unknown_period_type:,}",
        "",
        "说明：固定披露滞后规则不是提前可得规则；它会牺牲一部分及时性来降低财报信息泄漏风险。若 `available_date - state_date` 大量偏大，通常来自 HMM 日度状态样本未覆盖可得日附近交易日，需要后续扩展 HMM 日度状态。",
        "",
        "## 市场状态样本数量",
        "",
        markdown_table(regime_counts),
        "",
        "## 各市场状态下 Gap 分布",
        "",
        markdown_table(gap_regime),
        "",
        "## EntryScore 分位数组下 Gap 均值",
        "",
        markdown_table(entry_quantile),
        "",
        "## Gap 与市场状态变量相关性",
        "",
        markdown_table(corr),
        "",
        f"## bullish_transition_signal=1 样本数量",
        "",
        f"- {bullish_count:,}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_integration_report(
    path: Path,
    panel: pd.DataFrame,
    gap_regime: pd.DataFrame,
    entry_quantile: pd.DataFrame,
    corr: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    matched_rows = int(panel["state_date"].notna().sum())
    total_rows = len(panel)
    state_after_available = int((panel["state_date"] > panel["available_date"]).sum())
    high_corr = corr.loc[corr["abs_corr_gt_0_70"].astype(bool)]
    interaction_coverage = {
        col: {
            "non_null_rows": int(panel[col].notna().sum()),
            "nonzero_rows": int((panel[col].fillna(0) != 0).sum()),
        }
        for col in INTERACTION_COLUMNS
        if col in panel
    }
    data_gaps = [
        "最优负债率主结果未包含公告日期字段，主对齐使用规则化可得日。",
        "当前任务未接入企业未来收益或未来风险标签，因此只生成后续检验所需解释变量和交互变量。",
    ]
    if int((panel["available_to_state_days"] > 30).sum()) > 0:
        data_gaps.append(
            "部分样本的可得日与最近 HMM 状态日相隔超过 30 天，后续正式检验前建议扩展 HMM 日度状态覆盖。"
        )

    lines = [
        "# Gap × Market State 集成报告",
        "",
        "## 1. 主输出文件",
        "",
        f"- 最优负债率模块：`{LEVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- HMM 市场状态模块：`{HMM_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 未使用平滑概率文件：`{SMOOTHED_DIAGNOSTIC_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## 2. 对齐方式",
        "",
        "本次同时实现了两种对齐方式：方式 A 使用报告期截止日；方式 B 使用报告期后可得日。主面板采用方式 B。由于最优负债率主文件没有公告日期字段，方式 B 使用规则化可得日：季报和半年报加 45 天，年报加 120 天。",
        "",
        "## 3. 未来信息泄漏检查",
        "",
        f"- 主面板 `state_date > available_date` 行数：{state_after_available:,}",
        f"- 正式 HMM 输入文件为 forward filtering 概率文件，未读取 smoothing 诊断文件。",
        "- 规则化可得日避免把尚未公开的财报信息与报告期当天市场状态强行绑定；但最新报告期若 HMM 未覆盖到可得日附近，会被标记为状态日期偏旧。",
        "",
        "## 4. 匹配样本",
        "",
        f"- 企业-期间总样本：{total_rows:,}",
        f"- 成功匹配市场状态：{matched_rows:,}",
        f"- 未匹配：{total_rows - matched_rows:,}",
        "",
        "## 5. 三个市场状态下 Gap 分布",
        "",
        markdown_table(gap_regime),
        "",
        "## 6. EntryScore 与 Gap 相关性",
        "",
        markdown_table(corr[corr["right"].eq("entry_score")]),
        "",
        "高相关性阈值采用绝对相关系数 0.70。",
        (
            "未发现超过阈值的相关性。"
            if high_corr.empty
            else "发现超过阈值的相关性："
        ),
    ]
    if not high_corr.empty:
        lines.extend(["", markdown_table(high_corr)])

    lines.extend(
        [
            "",
            "## 7. 交互变量生成情况",
            "",
            markdown_table(
                pd.DataFrame(
                    [
                        {
                            "variable": key,
                            "non_null_rows": value["non_null_rows"],
                            "nonzero_rows": value["nonzero_rows"],
                        }
                        for key, value in interaction_coverage.items()
                    ]
                )
            ),
            "",
            "## 8. 可进入下一步检验的结果",
            "",
            "- 主解释变量：`leverage_gap`。",
            "- 市场状态变量：`entry_score`、`p_low_bull`、`p_high_entropy`、`p_low_bear`、`market_regime`、`bullish_transition_signal`、`bearish_transition_signal`。",
            "- 交互项：`gap_x_entry_score`、`gap_x_p_low_bull`、`gap_x_p_high_entropy`、`gap_x_p_low_bear`、`gap_x_bullish_transition`、`gap_x_H`、`gap_x_Lplus`、`gap_x_Lminus`。",
            "",
            "## 9. 数据缺口",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in data_gaps])
    lines.extend(
        [
            "",
            "## 10. 后续检验接口",
            "",
            "面板已经保留 `firm_id`、`period_date`、`available_date`、`state_date` 和全部 Gap × MarketState 交互项，可直接与企业未来风险或未来收益标签按企业和未来窗口合并后估计风险/收益方向性检验。当前未强行估计正式资产定价或收益回归。",
            "",
            "## EntryScore 分组描述",
            "",
            markdown_table(entry_quantile),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    leverage, hmm, metadata = load_inputs()
    leverage = infer_available_date(leverage, metadata["announcement_cols"])

    panel_a = merge_asof_state(
        leverage,
        hmm,
        alignment_date_col="period_date",
        alignment_method="period_end_cutoff",
    )
    panel_a["available_date"] = panel_a["period_date"]
    panel_a["available_date_source"] = "period_date_for_alignment_A"
    panel_a = add_constructed_variables(panel_a)

    panel_b = merge_asof_state(
        leverage,
        hmm,
        alignment_date_col="available_date",
        alignment_method="available_date_rule_45_120",
    )
    panel_b = add_constructed_variables(panel_b)

    diag_a = diagnostics(panel_a, hmm, "period_end_cutoff", "period_date")
    diag_b = diagnostics(panel_b, hmm, "available_date_rule_45_120", "available_date")

    main_panel = panel_b[PANEL_COLUMNS].copy()
    main_panel = main_panel.sort_values(["firm_id", "period_date", "state_date"])

    gap_regime = gap_by_regime(main_panel)
    entry_quantile = entry_score_quantile_gap(main_panel)
    corr = correlation_summary(main_panel)
    summary_stats = numeric_summary(main_panel, SUMMARY_NUMERIC_COLUMNS)
    interaction_stats = numeric_summary(main_panel, INTERACTION_COLUMNS)

    main_panel.to_csv(
        OUTPUT_DIR / "integrated_gap_market_state_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_panel.to_parquet(
        OUTPUT_DIR / "integrated_gap_market_state_panel.parquet",
        index=False,
    )
    summary_stats.to_csv(
        OUTPUT_DIR / "integration_summary_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gap_regime.to_csv(
        OUTPUT_DIR / "gap_by_market_regime.csv",
        index=False,
        encoding="utf-8-sig",
    )
    interaction_stats.to_csv(
        OUTPUT_DIR / "interaction_variable_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_alignment_report(
        OUTPUT_DIR / "alignment_report.md",
        leverage,
        hmm,
        panel_a,
        panel_b,
        diag_a,
        diag_b,
        gap_regime,
        entry_quantile,
        corr,
        metadata,
    )
    write_integration_report(
        OUTPUT_DIR / "integration_report.md",
        main_panel,
        gap_regime,
        entry_quantile,
        corr,
        metadata,
    )

    print("Wrote integrated panel and reports to", OUTPUT_DIR)
    print(f"Rows: {len(main_panel):,}; matched: {main_panel['state_date'].notna().sum():,}")
    print(
        "Leak check state_date > available_date:",
        int((main_panel["state_date"] > main_panel["available_date"]).sum()),
    )
    print("Used smoothed probabilities:", metadata["used_smoothed_probabilities"])


if __name__ == "__main__":
    main()
