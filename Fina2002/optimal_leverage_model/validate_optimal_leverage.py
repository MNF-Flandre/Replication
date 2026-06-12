from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAWDATA_ROOT = Path(os.environ.get("QUANT_RAWDATA_ROOT", PROJECT_ROOT / "external_data")).expanduser()
NEAR_OPTIMAL_TOL = 0.02
DEBT_RATIO_UPPER = 1.0


def finite_mask(*series: pd.Series) -> pd.Series:
    if not series:
        return pd.Series(dtype=bool)
    mask = pd.Series(True, index=series[0].index)
    for item in series:
        mask &= item.notna() & np.isfinite(item)
    return mask


def distribution_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        rows.append(
            {
                "variable": col,
                "count": int(s.notna().sum()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "p1": float(s.quantile(0.01)) if s.notna().any() else np.nan,
                "p5": float(s.quantile(0.05)) if s.notna().any() else np.nan,
                "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
                "median": float(s.quantile(0.50)) if s.notna().any() else np.nan,
                "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
                "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
                "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(empty)"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    formatted = view.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        elif pd.api.types.is_integer_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(int(x)))
        else:
            formatted[col] = formatted[col].astype(str).replace("nan", "")
    lines = ["| " + " | ".join(map(str, formatted.columns)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(formatted.columns)) + " |")
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in formatted.columns) + " |")
    return "\n".join(lines)


def ols_with_intercept(y: pd.Series, x: pd.Series) -> dict[str, Any]:
    mask = finite_mask(y, x)
    yy = y.loc[mask].astype(float).to_numpy()
    xx = x.loc[mask].astype(float).to_numpy()
    n = int(len(yy))
    result: dict[str, Any] = {
        "n_obs": n,
        "alpha": np.nan,
        "beta": np.nan,
        "beta_se": np.nan,
        "beta_t": np.nan,
        "r2": np.nan,
        "mean_y": float(np.mean(yy)) if n else np.nan,
    }
    if n < 3:
        return result
    X = np.column_stack([np.ones(n), xx])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ coef
    sse = float(np.sum(resid**2))
    sst = float(np.sum((yy - yy.mean()) ** 2))
    dof = n - X.shape[1]
    sigma2 = sse / dof if dof > 0 else np.nan
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2) if np.isfinite(sigma2) else np.array([np.nan, np.nan])
    result.update(
        {
            "alpha": float(coef[0]),
            "beta": float(coef[1]),
            "beta_se": float(se[1]),
            "beta_t": float(coef[1] / se[1]) if se[1] and np.isfinite(se[1]) else np.nan,
            "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        }
    )
    return result


def no_intercept_calibration(df: pd.DataFrame, c_col: str = "C_core") -> dict[str, Any]:
    mask = finite_mask(df["future_delta_d_annualized"], df[c_col], df["observed_debt_ratio"])
    reg = df.loc[mask, ["future_delta_d_annualized", c_col, "observed_debt_ratio"]]
    n_obs = int(len(reg))
    result: dict[str, Any] = {
        "n_obs": n_obs,
        "b_C": np.nan,
        "b_d": np.nan,
        "kappa_hat": np.nan,
        "s_hat": np.nan,
        "phi0_hat": np.nan,
        "calibration_valid": False,
        "reason_if_invalid": "",
    }
    if n_obs < 30:
        result["reason_if_invalid"] = f"n_obs={n_obs} < 30"
        return result
    X = reg[[c_col, "observed_debt_ratio"]].to_numpy(dtype=float)
    y = reg["future_delta_d_annualized"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b_c = float(coef[0])
    b_d = float(coef[1])
    result["b_C"] = b_c
    result["b_d"] = b_d
    if not np.isfinite(b_c) or not np.isfinite(b_d):
        result["reason_if_invalid"] = "non-finite regression coefficient"
    elif b_c <= 0:
        result["reason_if_invalid"] = "b_C <= 0"
    elif b_d >= 0:
        result["reason_if_invalid"] = "b_d >= 0"
    else:
        result["kappa_hat"] = -b_d
        result["s_hat"] = b_c / (-b_d)
        result["calibration_valid"] = True
    return result


def compute_core(tax_rate: pd.Series, debt_cost: pd.Series, eta: float, horizon: float) -> pd.Series:
    a = 1.0 / (eta - 1.0)
    numerator = tax_rate * ((1.0 + debt_cost) ** horizon - 1.0)
    valid = tax_rate.notna() & debt_cost.notna() & (debt_cost > -1.0) & (numerator >= 0) & np.isfinite(numerator)
    return pd.Series(np.where(valid, (numerator / (horizon * eta)) ** a, np.nan), index=tax_rate.index)


def compute_target(tax_rate: pd.Series, debt_cost: pd.Series, eta: float, horizon: float, phi0: float) -> pd.Series:
    a = 1.0 / (eta - 1.0)
    numerator = tax_rate * ((1.0 + debt_cost) ** horizon - 1.0)
    denominator = phi0 * horizon * eta
    valid = tax_rate.notna() & debt_cost.notna() & (debt_cost > -1.0) & (numerator >= 0) & np.isfinite(numerator) & (denominator > 0)
    raw = pd.Series(np.where(valid, (numerator / denominator) ** a, np.nan), index=tax_rate.index)
    return raw.clip(lower=0.0, upper=DEBT_RATIO_UPPER)


def add_next_changes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["firm_id", "period_date"]).copy()
    out["next_observed_debt_ratio"] = out.groupby("firm_id")["observed_debt_ratio"].shift(-1)
    out["next_period_date"] = out.groupby("firm_id")["period_date"].shift(-1)
    out["delta_years_to_next"] = (out["next_period_date"] - out["period_date"]).dt.days / 365.25
    valid = (
        out["next_observed_debt_ratio"].notna()
        & out["delta_years_to_next"].notna()
        & (out["delta_years_to_next"] > 0)
        & (out["delta_years_to_next"] <= 1.1)
        & out["next_observed_debt_ratio"].between(0, 1, inclusive="neither")
    )
    out["future_delta_d"] = (out["next_observed_debt_ratio"] - out["observed_debt_ratio"]).where(valid)
    out["future_delta_d_annualized"] = (out["future_delta_d"] / out["delta_years_to_next"]).where(valid)
    return out


def load_industry_map(rawdata_root: Path) -> pd.DataFrame:
    path = rawdata_root / "audit" / "RESSET_CINFO_1.csv"
    if not path.exists():
        return pd.DataFrame(columns=["firm_id", "industry_code", "industry_name", "industry_section_code", "industry_section_name"])
    required = [
        "A股股票代码_A_StkCd",
        "证监会行业代码_CsrcICCd",
        "证监会行业名称_CsrcICNm",
        "证监会行业门类代码_CsrcIcCd1",
        "证监会行业门类名称_CsrcIcNm1",
    ]
    try:
        df = pd.read_csv(path, encoding="gb18030", usecols=required, low_memory=False)
    except pd.errors.ParserError:
        df = pd.read_csv(path, encoding="gb18030", usecols=required, engine="python", on_bad_lines="skip")
    if not set(required).issubset(df.columns):
        return pd.DataFrame(columns=["firm_id", "industry_code", "industry_name", "industry_section_code", "industry_section_name"])
    out = df[required].copy()
    out = out.rename(
        columns={
            "A股股票代码_A_StkCd": "firm_id",
            "证监会行业代码_CsrcICCd": "industry_code",
            "证监会行业名称_CsrcICNm": "industry_name",
            "证监会行业门类代码_CsrcIcCd1": "industry_section_code",
            "证监会行业门类名称_CsrcIcNm1": "industry_section_name",
        }
    )
    out["firm_id"] = pd.to_numeric(out["firm_id"], errors="coerce")
    out = out.dropna(subset=["firm_id"]).copy()
    out["firm_id"] = out["firm_id"].astype(int)
    out = out.drop_duplicates("firm_id", keep="first")
    return out


def leverage_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    labels = ["(0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", "(0.4,0.5]", "(0.5,1]"]
    work = df.copy()
    work["observed_debt_ratio_bin"] = pd.cut(work["observed_debt_ratio"], bins=bins, labels=labels, include_lowest=False)
    table = (
        work.groupby("observed_debt_ratio_bin", observed=True)
        .agg(
            n=("optimal_debt_ratio", "size"),
            observed_debt_ratio_mean=("observed_debt_ratio", "mean"),
            optimal_debt_ratio_mean=("optimal_debt_ratio", "mean"),
            optimal_debt_ratio_median=("optimal_debt_ratio", "median"),
            leverage_gap_mean=("leverage_gap", "mean"),
            leverage_gap_median=("leverage_gap", "median"),
        )
        .reset_index()
    )
    return table


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["tax_rate", "debt_cost", "observed_debt_ratio"]:
        mask = finite_mask(df["optimal_debt_ratio"], df[var])
        rows.append(
            {
                "x": var,
                "y": "optimal_debt_ratio",
                "n": int(mask.sum()),
                "pearson_corr": float(df.loc[mask, ["optimal_debt_ratio", var]].corr(method="pearson").iloc[0, 1]) if int(mask.sum()) > 2 else np.nan,
                "spearman_corr": float(df.loc[mask, ["optimal_debt_ratio", var]].corr(method="spearman").iloc[0, 1]) if int(mask.sum()) > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def dynamic_group_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[df["future_delta_d_annualized"].notna()].copy()
    status_order = ["under_levered", "near_optimal", "over_levered"]
    by_status = (
        valid.groupby("leverage_status")
        .agg(
            n=("future_delta_d_annualized", "size"),
            gap_mean=("leverage_gap", "mean"),
            gap_median=("leverage_gap", "median"),
            future_delta_d_mean=("future_delta_d", "mean"),
            future_delta_d_median=("future_delta_d", "median"),
            future_delta_d_annualized_mean=("future_delta_d_annualized", "mean"),
            future_delta_d_annualized_median=("future_delta_d_annualized", "median"),
            pct_future_debt_ratio_increase=("future_delta_d", lambda x: float((x > 0).mean())),
        )
        .reindex(status_order)
        .dropna(how="all")
        .reset_index()
    )
    tercile_valid = valid[valid["leverage_gap"].notna()].copy()
    tercile_valid["gap_tercile"] = pd.qcut(tercile_valid["leverage_gap"], 3, labels=["Low Gap", "Middle Gap", "High Gap"], duplicates="drop")
    by_tercile = (
        tercile_valid.groupby("gap_tercile", observed=True)
        .agg(
            n=("future_delta_d_annualized", "size"),
            gap_mean=("leverage_gap", "mean"),
            gap_median=("leverage_gap", "median"),
            future_delta_d_mean=("future_delta_d", "mean"),
            future_delta_d_median=("future_delta_d", "median"),
            future_delta_d_annualized_mean=("future_delta_d_annualized", "mean"),
            future_delta_d_annualized_median=("future_delta_d_annualized", "median"),
            pct_future_debt_ratio_increase=("future_delta_d", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    return by_status, by_tercile


def rolling_oos(df: pd.DataFrame, eta: float, horizon: float, min_history_obs: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    work["year"] = work["period_date"].dt.year
    work["next_year"] = work["next_period_date"].dt.year
    work["C_oos_base"] = compute_core(work["tax_rate"], work["debt_cost"], eta, horizon)
    out = work.copy()
    out["phi0_oos"] = np.nan
    out["optimal_debt_ratio_oos"] = np.nan
    out["leverage_gap_oos"] = np.nan
    out["leverage_status_oos"] = "missing"
    cal_rows: list[dict[str, Any]] = []
    for year in sorted(work["year"].dropna().unique()):
        history = work[(work["year"] <= year - 1) & (work["next_year"] <= year - 1)].copy()
        history["C_oos_base"] = compute_core(history["tax_rate"], history["debt_cost"], eta, horizon)
        cal = no_intercept_calibration(history, "C_oos_base")
        cal["year"] = int(year)
        cal["history_last_year"] = int(year - 1)
        cal["rows_current_year"] = int((work["year"] == year).sum())
        if cal["calibration_valid"]:
            phi0 = ((-cal["b_d"]) / cal["b_C"]) ** (eta - 1.0)
            if np.isfinite(phi0) and phi0 > 0 and cal["n_obs"] >= min_history_obs:
                cal["phi0_hat"] = float(phi0)
                current_idx = out.index[out["year"] == year]
                target = compute_target(out.loc[current_idx, "tax_rate"], out.loc[current_idx, "debt_cost"], eta, horizon, phi0)
                out.loc[current_idx, "phi0_oos"] = phi0
                out.loc[current_idx, "optimal_debt_ratio_oos"] = target
                out.loc[current_idx, "leverage_gap_oos"] = out.loc[current_idx, "observed_debt_ratio"] - target
                gap = out.loc[current_idx, "leverage_gap_oos"]
                out.loc[current_idx, "leverage_status_oos"] = np.select(
                    [gap.isna(), gap.abs() <= NEAR_OPTIMAL_TOL, gap > NEAR_OPTIMAL_TOL, gap < -NEAR_OPTIMAL_TOL],
                    ["missing", "near_optimal", "over_levered", "under_levered"],
                    default="missing",
                )
                cal["oos_dstar_median"] = float(target.median()) if target.notna().any() else np.nan
                cal["oos_gap_median"] = float((out.loc[current_idx, "observed_debt_ratio"] - target).median()) if target.notna().any() else np.nan
            else:
                cal["calibration_valid"] = False
                cal["reason_if_invalid"] = f"history n_obs={cal['n_obs']} below min_history_obs={min_history_obs} or invalid phi0"
        cal_rows.append(cal)
    cal_df = pd.DataFrame(cal_rows)
    oos_reg = ols_with_intercept(out["future_delta_d_annualized"], out["leverage_gap_oos"])
    oos_reg["model"] = "rolling_oos_gap"
    return out, cal_df, oos_reg


def eta_t_robustness(df: pd.DataFrame, etas: list[float], horizons: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for eta in etas:
        for horizon in horizons:
            c_col = f"C_eta_{eta}_T_{horizon}"
            work = df.copy()
            work[c_col] = compute_core(work["tax_rate"], work["debt_cost"], eta, horizon)
            cal = no_intercept_calibration(work, c_col)
            row = {
                "eta": eta,
                "T": horizon,
                **cal,
                "dstar_median": np.nan,
                "gap_median": np.nan,
                "over_levered_count": np.nan,
                "under_levered_count": np.nan,
                "near_optimal_count": np.nan,
                "dynamic_beta": np.nan,
                "dynamic_beta_t": np.nan,
                "dynamic_r2": np.nan,
            }
            if cal["calibration_valid"]:
                phi0 = ((-cal["b_d"]) / cal["b_C"]) ** (eta - 1.0)
                row["phi0_hat"] = float(phi0)
                target = compute_target(work["tax_rate"], work["debt_cost"], eta, horizon, phi0)
                gap = work["observed_debt_ratio"] - target
                status = np.select(
                    [gap.isna(), gap.abs() <= NEAR_OPTIMAL_TOL, gap > NEAR_OPTIMAL_TOL, gap < -NEAR_OPTIMAL_TOL],
                    ["missing", "near_optimal", "over_levered", "under_levered"],
                    default="missing",
                )
                vc = pd.Series(status).value_counts()
                dyn = ols_with_intercept(work["future_delta_d_annualized"], gap)
                row.update(
                    {
                        "dstar_median": float(target.median()),
                        "gap_median": float(gap.median()),
                        "over_levered_count": int(vc.get("over_levered", 0)),
                        "under_levered_count": int(vc.get("under_levered", 0)),
                        "near_optimal_count": int(vc.get("near_optimal", 0)),
                        "dynamic_beta": dyn["beta"],
                        "dynamic_beta_t": dyn["beta_t"],
                        "dynamic_r2": dyn["r2"],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    validation_dir: Path,
    main_df: pd.DataFrame,
    dist: pd.DataFrame,
    by_year: pd.DataFrame,
    by_industry: pd.DataFrame,
    corr: pd.DataFrame,
    dyn_reg: pd.DataFrame,
    dyn_status: pd.DataFrame,
    dyn_tercile: pd.DataFrame,
    oos_cal: pd.DataFrame,
    oos_reg: dict[str, Any],
    robust_grid: pd.DataFrame,
) -> None:
    corr_dobs = corr.loc[corr["x"].eq("observed_debt_ratio"), "pearson_corr"]
    corr_dobs_value = float(corr_dobs.iloc[0]) if not corr_dobs.empty else np.nan
    corr_warning = "是" if np.isfinite(corr_dobs_value) and abs(corr_dobs_value) > 0.8 else "否"
    tau_corr = float(corr.loc[corr["x"].eq("tax_rate"), "pearson_corr"].iloc[0])
    r_corr = float(corr.loc[corr["x"].eq("debt_cost"), "pearson_corr"].iloc[0])
    beta = float(dyn_reg.loc[0, "beta"]) if not dyn_reg.empty else np.nan
    beta_ok = "成立" if np.isfinite(beta) and beta < 0 else "不成立"
    valid_oos = oos_cal[oos_cal["phi0_hat"].notna()] if "phi0_hat" in oos_cal else pd.DataFrame()
    industry_match_rate = float(main_df["industry_name"].notna().mean()) if "industry_name" in main_df.columns else 0.0
    lines = [
        "# A_clean_main 最优负债率有效性验证报告",
        "",
        "## 主口径冻结",
        "",
        "本报告固定 `A_clean_main` 为主候选，不修改结构性 trade-off 主公式，也不加入市值、行业、年份、盈利能力等控制变量。以下检验只验证 `d*` 是否像合理目标负债率，以及 `Gap=d_obs-d*` 是否能预测后续资本结构调整。",
        "",
        "## 1. d* 经济合理性",
        "",
        "核心分布：",
        "",
        markdown_table(dist),
        "",
        "相关系数检验：",
        "",
        markdown_table(corr),
        "",
        f"- `corr(d*, tau)={tau_corr:.4f}`，税率方向为正。",
        f"- `corr(d*, r)={r_corr:.4f}`，债务成本方向为正。",
        f"- `corr(d*, d_obs)={corr_dobs_value:.4f}`，是否超过 0.8 警戒线：{corr_warning}。",
        "",
        "按年份的 `d*` 中位数已输出到 `A_dstar_by_year.csv`，前几行如下：",
        "",
        markdown_table(by_year.head(12)),
        "",
        "按行业的 `d*` 中位数已输出到 `A_dstar_by_industry.csv`，前几行如下：",
        "",
        f"行业字段来自 `RESSET_CINFO_1.csv` 的证监会行业分类，A 样本行匹配率为 {industry_match_rate:.2%}。",
        "",
        markdown_table(by_industry.head(12) if not by_industry.empty else by_industry),
        "",
        "按实际负债率分组：",
        "",
        markdown_table(pd.read_csv(validation_dir / "A_dstar_by_observed_leverage_bin.csv")),
        "",
        "## 2. 动态调整检验",
        "",
        "回归设定：`future_delta_d_annualized = alpha + beta * Gap + error`。理论预测是 `beta<0`。",
        "",
        markdown_table(dyn_reg),
        "",
        f"结论：`beta={beta:.6g}`，理论方向检验{beta_ok}。",
        "",
        "按杠杆状态分组的后续变化：",
        "",
        markdown_table(dyn_status),
        "",
        "按 Gap 三分位分组的后续变化：",
        "",
        markdown_table(dyn_tercile),
        "",
        "## 3. 样本外 rolling phi0",
        "",
        "样本外版本对每一年只使用上一年及以前已经可观察到的动态调整样本估计 `phi0`，再计算当年 `d*`，避免使用未来信息。",
        "",
        markdown_table(oos_cal[["year", "n_obs", "b_C", "b_d", "phi0_hat", "calibration_valid", "oos_dstar_median", "oos_gap_median", "reason_if_invalid"]].tail(15)),
        "",
        "rolling OOS Gap 的动态调整回归：",
        "",
        markdown_table(pd.DataFrame([oos_reg])),
        "",
        f"可用 rolling 年份数：{len(valid_oos)}；`phi0` 中位数：{valid_oos['phi0_hat'].median() if not valid_oos.empty else np.nan:.6g}。",
        "",
        "## 4. eta 与 T 稳健性",
        "",
        "每个参数组合都在冻结后的 A_clean_main 输出样本上重新估计 `phi0`，再计算 `d*`、`Gap` 和动态调整 beta。正式主候选仍以 `phi0_calibration_robust_variants.csv` 中的 A 版 `phi0=0.0267726` 为准。",
        "",
        markdown_table(robust_grid[["eta", "T", "n_obs", "b_C", "b_d", "phi0_hat", "dstar_median", "gap_median", "over_levered_count", "under_levered_count", "near_optimal_count", "dynamic_beta", "dynamic_beta_t"]]),
        "",
        "## 输出文件",
        "",
        f"- 验证目录：`{validation_dir}`",
        "- `A_distribution.csv`",
        "- `A_dstar_by_year.csv`",
        "- `A_dstar_by_industry.csv`",
        "- `A_dstar_by_observed_leverage_bin.csv`",
        "- `A_correlations.csv`",
        "- `A_dynamic_adjustment_regression.csv`",
        "- `A_dynamic_adjustment_by_status.csv`",
        "- `A_dynamic_adjustment_by_gap_tercile.csv`",
        "- `A_rolling_phi0_oos.csv`",
        "- `variant_A_rolling_oos_results.csv`",
        "- `eta_T_robustness.csv`",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate A_clean_main optimal leverage economic reasonableness and dynamics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rawdata-root", type=Path, default=DEFAULT_RAWDATA_ROOT)
    parser.add_argument("--min-oos-history-obs", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_dir = args.output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    main_path = args.output_dir / "variant_A_clean_main_results.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"Missing A_clean_main result: {main_path}")
    df = pd.read_csv(main_path)
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["firm_id"] = pd.to_numeric(df["firm_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["firm_id", "period_date"]).copy()
    df["firm_id"] = df["firm_id"].astype(int)
    df["year"] = df["period_date"].dt.year
    df = add_next_changes(df)

    industry = load_industry_map(args.rawdata_root)
    df = df.merge(industry, on="firm_id", how="left")

    dist = distribution_table(df, ["optimal_debt_ratio", "observed_debt_ratio", "leverage_gap"])
    dist.to_csv(validation_dir / "A_distribution.csv", index=False, encoding="utf-8-sig")

    by_year = (
        df.groupby("year")
        .agg(
            n=("optimal_debt_ratio", "size"),
            dstar_median=("optimal_debt_ratio", "median"),
            dstar_mean=("optimal_debt_ratio", "mean"),
            dobs_median=("observed_debt_ratio", "median"),
            gap_median=("leverage_gap", "median"),
        )
        .reset_index()
    )
    by_year.to_csv(validation_dir / "A_dstar_by_year.csv", index=False, encoding="utf-8-sig")

    if "industry_name" in df.columns and df["industry_name"].notna().any():
        by_industry = (
            df.groupby(["industry_section_code", "industry_section_name", "industry_code", "industry_name"], dropna=False)
            .agg(
                n=("optimal_debt_ratio", "size"),
                firms=("firm_id", "nunique"),
                dstar_median=("optimal_debt_ratio", "median"),
                dstar_mean=("optimal_debt_ratio", "mean"),
                dobs_median=("observed_debt_ratio", "median"),
                gap_median=("leverage_gap", "median"),
            )
            .reset_index()
            .sort_values(["industry_section_code", "industry_code"])
        )
    else:
        by_industry = pd.DataFrame(columns=["industry_section_code", "industry_section_name", "industry_code", "industry_name", "n", "firms", "dstar_median", "dstar_mean", "dobs_median", "gap_median"])
    by_industry.to_csv(validation_dir / "A_dstar_by_industry.csv", index=False, encoding="utf-8-sig")

    by_bin = leverage_bin_table(df)
    by_bin.to_csv(validation_dir / "A_dstar_by_observed_leverage_bin.csv", index=False, encoding="utf-8-sig")

    corr = correlation_table(df)
    corr.to_csv(validation_dir / "A_correlations.csv", index=False, encoding="utf-8-sig")

    dyn = ols_with_intercept(df["future_delta_d_annualized"], df["leverage_gap"])
    dyn["model"] = "A_clean_main_gap"
    dyn_reg = pd.DataFrame([dyn])
    dyn_reg = dyn_reg[["model", "n_obs", "alpha", "beta", "beta_se", "beta_t", "r2", "mean_y"]]
    dyn_reg.to_csv(validation_dir / "A_dynamic_adjustment_regression.csv", index=False, encoding="utf-8-sig")

    dyn_status, dyn_tercile = dynamic_group_tables(df)
    dyn_status.to_csv(validation_dir / "A_dynamic_adjustment_by_status.csv", index=False, encoding="utf-8-sig")
    dyn_tercile.to_csv(validation_dir / "A_dynamic_adjustment_by_gap_tercile.csv", index=False, encoding="utf-8-sig")

    oos_result, oos_cal, oos_reg = rolling_oos(df, eta=2.0, horizon=1.0, min_history_obs=args.min_oos_history_obs)
    oos_keep = [
        "model_variant",
        "firm_id",
        "period_date",
        "year",
        "observed_debt_ratio",
        "tax_rate",
        "debt_cost",
        "optimal_debt_ratio",
        "leverage_gap",
        "phi0_oos",
        "optimal_debt_ratio_oos",
        "leverage_gap_oos",
        "leverage_status_oos",
        "future_delta_d",
        "future_delta_d_annualized",
    ]
    oos_result[[col for col in oos_keep if col in oos_result.columns]].to_csv(validation_dir / "variant_A_rolling_oos_results.csv", index=False, encoding="utf-8-sig")
    oos_cal.to_csv(validation_dir / "A_rolling_phi0_oos.csv", index=False, encoding="utf-8-sig")

    robust_grid = eta_t_robustness(df, etas=[1.5, 2.0, 3.0], horizons=[1.0, 2.0, 3.0])
    robust_grid.to_csv(validation_dir / "eta_T_robustness.csv", index=False, encoding="utf-8-sig")

    abc_path = args.output_dir / "phi0_calibration_robust_variants.csv"
    if abc_path.exists():
        abc = pd.read_csv(abc_path)
        abc.to_csv(validation_dir / "ABC_frozen_A_comparison.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "main_candidate": "A_clean_main",
        "main_result_path": str(main_path),
        "n_rows": int(len(df)),
        "n_firms": int(df["firm_id"].nunique()),
        "n_years": int(df["year"].nunique()),
        "industry_matched_rows": int(df["industry_name"].notna().sum()) if "industry_name" in df.columns else 0,
        "industry_match_rate": float(df["industry_name"].notna().mean()) if "industry_name" in df.columns else 0.0,
    }
    (validation_dir / "validation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(
        args.output_dir,
        validation_dir,
        df,
        dist,
        by_year,
        by_industry,
        corr,
        dyn_reg,
        dyn_status,
        dyn_tercile,
        oos_cal,
        oos_reg,
        robust_grid,
    )

    print(f"A_clean_main rows={len(df)}, firms={df['firm_id'].nunique()}, years={df['year'].nunique()}")
    print(f"Industry match rate={manifest['industry_match_rate']:.2%}")
    print(f"Dynamic beta={dyn['beta']:.6g}, t={dyn['beta_t']:.6g}, n={dyn['n_obs']}")
    print(f"Validation report: {args.output_dir / 'validation_report.md'}")
    print(f"Validation tables: {validation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
