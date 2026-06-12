from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from validate_optimal_leverage import markdown_table


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_EVENT_PANEL = DEFAULT_OUTPUT_DIR / "gap_return_test" / "gap_return_event_panel.csv"
HORIZONS = [21, 63, 126]
KEY_TERMS = ["gap_10pp", "dobs_10pp", "dstar_10pp", "unused_capacity_10pp", "over_excess_10pp"]


def add_choice_space_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal_trade_date"] = pd.to_datetime(out["signal_trade_date"], errors="coerce")
    out["signal_year"] = out["signal_trade_date"].dt.year
    out["industry_section_code"] = out["industry_section_code"].fillna("UNKNOWN").astype(str)
    out["industry_section_name"] = out["industry_section_name"].fillna("UNKNOWN").astype(str)
    out["industry_year"] = out["industry_section_code"] + "_" + out["signal_year"].astype("Int64").astype(str)
    out["gap_10pp"] = out["leverage_gap"] / 0.10
    out["dobs_10pp"] = out["observed_debt_ratio"] / 0.10
    out["dstar_10pp"] = out["optimal_debt_ratio"] / 0.10
    out["unused_debt_capacity"] = (out["optimal_debt_ratio"] - out["observed_debt_ratio"]).clip(lower=0.0)
    out["over_leverage_excess"] = (out["observed_debt_ratio"] - out["optimal_debt_ratio"]).clip(lower=0.0)
    out["unused_capacity_10pp"] = out["unused_debt_capacity"] / 0.10
    out["over_excess_10pp"] = out["over_leverage_excess"] / 0.10
    return out


def run_one_regression(df: pd.DataFrame, y_col: str, spec_name: str, formula_rhs: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    needed = ["firm_id", y_col, "industry_section_code", "signal_year", *KEY_TERMS]
    work = df.copy()
    work = work[(work["industry_section_code"] != "UNKNOWN") & work["signal_year"].notna()].copy()
    for col in needed:
        if col in work.columns:
            work = work[work[col].notna()]
    work = work[np.isfinite(work[y_col])].copy()
    formula = f"{y_col} ~ {formula_rhs}"
    model = smf.ols(formula, data=work)
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": work["firm_id"]})
    coef_rows: list[dict[str, Any]] = []
    for term in KEY_TERMS:
        if term in fitted.params.index:
            coef_rows.append(
                {
                    "spec": spec_name,
                    "horizon_days": int(y_col.replace("ret_", "").replace("d", "")),
                    "term": term,
                    "coef": float(fitted.params[term]),
                    "std_err": float(fitted.bse[term]),
                    "t": float(fitted.tvalues[term]),
                    "p_value": float(fitted.pvalues[term]),
                    "n_obs": int(fitted.nobs),
                    "r2": float(fitted.rsquared),
                }
            )
    model_row = {
        "spec": spec_name,
        "horizon_days": int(y_col.replace("ret_", "").replace("d", "")),
        "formula": formula,
        "n_obs": int(fitted.nobs),
        "r2": float(fitted.rsquared),
        "n_firms": int(work["firm_id"].nunique()),
        "n_industries": int(work["industry_section_code"].nunique()),
        "n_years": int(work["signal_year"].nunique()),
    }
    return coef_rows, model_row


def run_regressions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "gap_only_industry_year_FE": "gap_10pp + C(industry_section_code) + C(signal_year)",
        "gap_plus_dobs_industry_year_FE": "gap_10pp + dobs_10pp + C(industry_section_code) + C(signal_year)",
        "gap_plus_dstar_industry_year_FE": "gap_10pp + dstar_10pp + C(industry_section_code) + C(signal_year)",
        "choice_space_industry_year_FE": "unused_capacity_10pp + over_excess_10pp + dobs_10pp + C(industry_section_code) + C(signal_year)",
        "gap_plus_dobs_industryXyear_FE": "gap_10pp + dobs_10pp + C(industry_year)",
        "choice_space_industryXyear_FE": "unused_capacity_10pp + over_excess_10pp + dobs_10pp + C(industry_year)",
    }
    coef_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        y_col = f"ret_{horizon}d"
        for spec_name, rhs in specs.items():
            rows, model_row = run_one_regression(df, y_col, spec_name, rhs)
            coef_rows.extend(rows)
            model_rows.append(model_row)
    return pd.DataFrame(coef_rows), pd.DataFrame(model_rows)


def choice_space_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status, grp in df.groupby("leverage_status", dropna=False):
        rows.append(
            {
                "leverage_status": status,
                "n": int(len(grp)),
                "dobs_mean": float(grp["observed_debt_ratio"].mean()),
                "dstar_mean": float(grp["optimal_debt_ratio"].mean()),
                "gap_mean": float(grp["leverage_gap"].mean()),
                "unused_debt_capacity_mean": float(grp["unused_debt_capacity"].mean()),
                "unused_debt_capacity_median": float(grp["unused_debt_capacity"].median()),
                "over_leverage_excess_mean": float(grp["over_leverage_excess"].mean()),
                "over_leverage_excess_median": float(grp["over_leverage_excess"].median()),
            }
        )
    return pd.DataFrame(rows)


def compact_key_table(coefs: pd.DataFrame, spec_filter: str, terms: list[str]) -> pd.DataFrame:
    view = coefs[coefs["spec"].eq(spec_filter) & coefs["term"].isin(terms)].copy()
    view["coef_pct"] = view["coef"] * 100.0
    view["std_err_pct"] = view["std_err"] * 100.0
    return view[["horizon_days", "term", "coef_pct", "std_err_pct", "t", "p_value", "n_obs", "r2"]]


def write_report(output_dir: Path, result_dir: Path, coefs: pd.DataFrame, models: pd.DataFrame, space_summary: pd.DataFrame) -> None:
    main = compact_key_table(coefs, "gap_plus_dobs_industry_year_FE", ["gap_10pp", "dobs_10pp"])
    main_ixy = compact_key_table(coefs, "gap_plus_dobs_industryXyear_FE", ["gap_10pp", "dobs_10pp"])
    space = compact_key_table(coefs, "choice_space_industry_year_FE", ["unused_capacity_10pp", "over_excess_10pp", "dobs_10pp"])
    lines = [
        "# Gap 收益回归检验",
        "",
        "## 回归口径",
        "",
        "- 使用 `gap_return_event_panel.csv`，即 A_clean_main Gap 与披露后收益合并后的事件面板。",
        "- 信号日为 `Annodt` 之后第一个交易日，不使用报告截止日作为可交易时间。",
        "- 收益窗口为披露后 21 / 63 / 126 个交易日原始复合收益。",
        "- 标准误按企业 `firm_id` 聚类。",
        "- 主要固定效应为行业门类 FE + 披露年份 FE；另给出更严格的行业门类 × 披露年份 FE。",
        "",
        "注意：`Gap=d_obs-d*`，所以 `Gap`、`d_obs`、`d*` 三者不能同时放入一个回归，否则完全共线。为了检验 Gap 是否只是普通低杠杆效应，主规格使用 `Gap + d_obs`。",
        "",
        "所有斜率变量都按 10 个百分点缩放，因此系数表示变量增加 10 个百分点对应的收益变化，表中以百分比收益点显示。",
        "",
        "## 主检验：Gap + d_obs + 行业 FE + 年份 FE",
        "",
        markdown_table(main),
        "",
        "解释：如果 `gap_10pp` 系数为负，表示在控制实际负债率后，更高的相对过度负债偏离对应更低的后续收益。",
        "",
        "## 更严格固定效应：行业 × 年份 FE",
        "",
        markdown_table(main_ixy),
        "",
        "## 选择空间拆分",
        "",
        "定义：",
        "",
        "- `unused_debt_capacity = max(d* - d_obs, 0)`：企业低于目标负债率的可加杠杆空间。",
        "- `over_leverage_excess = max(d_obs - d*, 0)`：企业超过目标负债率的超额负债压力。",
        "- 两者都按 10 个百分点缩放进入回归。",
        "",
        "按 leverage_status 的选择空间分布：",
        "",
        markdown_table(space_summary),
        "",
        "选择空间回归：",
        "",
        markdown_table(space),
        "",
        "解释：`unused_capacity_10pp` 为正表示可加杠杆空间越大，后续收益越高；`over_excess_10pp` 为负表示超过目标的负债压力越大，后续收益越低。",
        "",
        "## 输出文件",
        "",
        f"- 回归系数：`{result_dir / 'gap_return_regression_coefficients.csv'}`",
        f"- 回归模型摘要：`{result_dir / 'gap_return_regression_models.csv'}`",
        f"- 选择空间分布：`{result_dir / 'gap_choice_space_summary.csv'}`",
    ]
    (output_dir / "gap_return_regression_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-disclosure return regressions on leverage Gap and choice-space variables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-panel", type=Path, default=DEFAULT_EVENT_PANEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.output_dir / "gap_return_test"
    result_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.event_panel)
    df = add_choice_space_variables(df)
    coefs, models = run_regressions(df)
    space_summary = choice_space_summary(df)
    coefs.to_csv(result_dir / "gap_return_regression_coefficients.csv", index=False, encoding="utf-8-sig")
    models.to_csv(result_dir / "gap_return_regression_models.csv", index=False, encoding="utf-8-sig")
    space_summary.to_csv(result_dir / "gap_choice_space_summary.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir, result_dir, coefs, models, space_summary)

    main = coefs[(coefs["spec"].eq("gap_plus_dobs_industry_year_FE")) & (coefs["term"].eq("gap_10pp"))]
    print("Main Gap coefficients, return percentage points per +10pp Gap:")
    for _, row in main.iterrows():
        print(f"{int(row['horizon_days'])}d: coef={row['coef']*100:.4f}pp, t={row['t']:.3f}, n={int(row['n_obs'])}")
    print(f"Report: {args.output_dir / 'gap_return_regression_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
