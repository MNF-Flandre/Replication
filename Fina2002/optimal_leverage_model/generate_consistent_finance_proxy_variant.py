from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BALANCE_PATH = Path(
    os.environ.get("QUANT_BALANCE_PATH", PROJECT_ROOT / "external_data" / "3tables" / "FS_Combas.csv")
).expanduser()
INCOME_PATH = Path(
    os.environ.get("QUANT_INCOME_PATH", PROJECT_ROOT / "external_data" / "3tables" / "FS_Comins.csv")
).expanduser()
FILENAME = "variant_D_finance_proxy_consistent_results.csv"

VARIANT = "D_finance_proxy_consistent"
LABEL = "Robustness D: consistently use finance expense B001211000 as the interest-expense proxy"
ETA = 2.0
TARGET_HORIZON_YEARS = 1.0
R_UPPER = 0.30
TAU_UPPER = 0.50
DEBT_RATIO_UPPER = 1.0
NEAR_OPTIMAL_TOL = 0.02
MIN_CALIBRATION_OBS = 30
STANDARD_MONTH_DAYS = {"03-31": 0.25, "06-30": 0.50, "09-30": 0.75, "12-31": 1.00}


def normalize_firm_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def read_source(path: Path, rename: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(rename), dtype={"Stkcd": "string"}, low_memory=False)
    df = df.rename(columns=rename)
    df["firm_id"] = normalize_firm_id(df["firm_id"])
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    if "report_scope" in df.columns:
        df["report_scope"] = df["report_scope"].astype("string").str.strip().str.upper()
        df = df.loc[df["report_scope"].eq("A")].copy()
    df = df.dropna(subset=["firm_id", "period_date"])
    value_cols = [col for col in df.columns if col not in {"firm_id", "period_date", "report_scope"}]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.groupby(["firm_id", "period_date"], as_index=False, dropna=False).first()


def build_panel() -> pd.DataFrame:
    balance = read_source(
        BALANCE_PATH,
        {
            "Stkcd": "firm_id",
            "Accper": "period_date",
            "Typrep": "report_scope",
            "A001000000": "total_assets",
            "A002000000": "total_liabilities",
            "A002101000": "short_term_debt",
            "A002201000": "long_term_debt",
            "A002203000": "bonds_payable",
            "A002107000": "notes_payable",
            "A002125000": "current_portion_long_term_debt",
            "A002211000": "lease_liabilities",
        },
    )
    income = read_source(
        INCOME_PATH,
        {
            "Stkcd": "firm_id",
            "Accper": "period_date",
            "Typrep": "report_scope",
            "B002100000": "tax_expense",
            "B001000000": "pretax_income",
            "B001211101": "direct_interest_expense",
            "B001211000": "finance_expense",
        },
    )
    panel = balance.merge(income, on=["firm_id", "period_date"], how="outer")
    panel = panel.sort_values(["firm_id", "period_date"]).reset_index(drop=True)
    month_day = panel["period_date"].dt.strftime("%m-%d")
    panel = panel.loc[month_day.isin(STANDARD_MONTH_DAYS)].copy()
    panel["is_standard_report_date"] = True
    return panel.reset_index(drop=True)


def append_flag(flags: pd.Series, mask: pd.Series | np.ndarray, flag: str) -> pd.Series:
    mask = pd.Series(mask, index=flags.index).fillna(False)
    flags.loc[mask] = np.where(flags.loc[mask].eq(""), flag, flags.loc[mask] + ";" + flag)
    return flags


def compute_debt(panel: pd.DataFrame, flags: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    out = panel.copy()
    component_cols = [
        "short_term_debt",
        "long_term_debt",
        "bonds_payable",
        "notes_payable",
        "current_portion_long_term_debt",
        "lease_liabilities",
    ]
    total_debt = out[component_cols].sum(axis=1, min_count=1)
    debt_source = pd.Series(np.where(total_debt.notna(), "sum_interest_bearing_debt_components", ""), index=out.index)

    fill_liabilities = total_debt.isna() & out["total_liabilities"].notna()
    total_debt.loc[fill_liabilities] = out.loc[fill_liabilities, "total_liabilities"]
    debt_source.loc[fill_liabilities] = "total_liabilities_proxy"
    flags = append_flag(flags, fill_liabilities, "observed_leverage_used_total_liabilities_proxy")

    out["total_debt_used"] = total_debt
    out["debt_source_flag"] = debt_source.replace("", np.nan)
    valid_observed = out["total_assets"].gt(0) & out["total_debt_used"].ge(0)
    out["observed_debt_ratio"] = (out["total_debt_used"] / out["total_assets"]).where(valid_observed)
    flags = append_flag(flags, out["total_assets"].isna(), "missing_total_assets")
    flags = append_flag(flags, out["total_assets"].notna() & out["total_assets"].le(0), "non_positive_total_assets")
    flags = append_flag(flags, out["total_debt_used"].isna(), "missing_debt_measure")
    flags = append_flag(flags, out["observed_debt_ratio"].notna() & out["observed_debt_ratio"].gt(1), "observed_debt_ratio_above_1")
    return out, flags


def compute_tax(panel: pd.DataFrame, flags: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    out = panel.copy()
    pretax = out["pretax_income"]
    raw = out["tax_expense"] / pretax
    non_positive_pretax = pretax.notna() & pretax.le(0)
    raw = raw.mask(non_positive_pretax, 0.0)
    clipped = raw.clip(lower=0.0, upper=1.0)
    out["tax_rate_raw"] = raw
    out["tax_rate"] = clipped
    flags = append_flag(flags, non_positive_pretax, "pretax_income_non_positive_tax_rate_set_to_0")
    flags = append_flag(flags, raw.notna() & raw.ne(clipped), "tax_rate_clipped_to_0_1")
    flags = append_flag(flags, clipped.isna(), "cannot_compute_tax_rate")
    return out, flags


def compute_interest_and_cost(panel: pd.DataFrame, flags: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    out = panel.sort_values(["firm_id", "period_date"]).copy()
    finance_interest = out["finance_expense"]
    out["interest_source_flag"] = np.where(finance_interest.notna(), "finance_expense_proxy", "missing")
    out["interest_expense_ytd_used"] = finance_interest
    out["fiscal_year"] = out["period_date"].dt.year
    out["fiscal_coverage_years"] = out["period_date"].dt.strftime("%m-%d").map(STANDARD_MONTH_DAYS).astype(float)

    prev_ytd = finance_interest.groupby([out["firm_id"], out["fiscal_year"]]).shift(1)
    prev_date = out.groupby(["firm_id", "fiscal_year"])["period_date"].shift(1)
    has_prev = prev_ytd.notna() & prev_date.notna()
    period_interest = finance_interest.where(~has_prev, finance_interest - prev_ytd)
    period_years = out["fiscal_coverage_years"].where(~has_prev, (out["period_date"] - prev_date).dt.days / 365.25)

    out["period_interest_expense"] = period_interest
    out["period_interest_years"] = period_years
    lag_debt = out.groupby("firm_id")["total_debt_used"].shift(1)
    avg_debt = (lag_debt + out["total_debt_used"]) / 2.0
    raw_period_cost = period_interest / avg_debt
    valid = avg_debt.gt(0) & period_years.gt(0) & raw_period_cost.notna()
    out["average_debt_for_cost"] = avg_debt
    out["debt_cost_raw"] = raw_period_cost.where(valid)
    out["debt_cost"] = (raw_period_cost / period_years).where(valid)

    flags = flags.loc[out.index].copy()
    flags = append_flag(flags, finance_interest.notna(), "finance_expense_used_as_interest_expense_proxy")
    flags = append_flag(flags, finance_interest.isna(), "missing_finance_expense_for_forced_proxy")
    flags = append_flag(
        flags,
        out["direct_interest_expense"].notna() & finance_interest.notna(),
        "direct_interest_expense_ignored_for_consistent_finance_proxy",
    )
    flags = append_flag(flags, period_interest.isna(), "missing_period_interest_expense")
    flags = append_flag(flags, period_interest.notna() & period_interest.lt(0), "negative_period_interest_expense")
    flags = append_flag(flags, period_years.isna() | period_years.le(0), "missing_period_length_for_debt_cost")
    flags = append_flag(flags, lag_debt.isna() | lag_debt.le(0) | avg_debt.le(0), "missing_or_non_positive_average_debt_for_debt_cost")
    flags = append_flag(flags, out["debt_cost"].isna(), "cannot_compute_debt_cost")
    flags = append_flag(flags, out["debt_cost"].notna() & out["debt_cost"].le(0), "non_positive_debt_cost")
    flags = append_flag(flags, out["debt_cost"].notna() & out["debt_cost"].ge(R_UPPER), f"debt_cost_outside_0_{R_UPPER:g}")
    flags = append_flag(flags, out["debt_cost"].notna() & out["debt_cost"].gt(1), "annualized_debt_cost_above_100pct")
    return out, flags


def compute_core(panel: pd.DataFrame, flags: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    out = panel.copy()
    a = 1.0 / (ETA - 1.0)
    numerator_core = out["tax_rate"] * ((1.0 + out["debt_cost"]) ** TARGET_HORIZON_YEARS - 1.0)
    valid_core = (
        out["tax_rate"].notna()
        & out["debt_cost"].notna()
        & out["debt_cost"].gt(-1)
        & numerator_core.ge(0)
        & np.isfinite(numerator_core)
    )
    out["target_horizon_years"] = TARGET_HORIZON_YEARS
    out["eta"] = ETA
    out["C_core"] = np.where(valid_core, (numerator_core / (TARGET_HORIZON_YEARS * ETA)) ** a, np.nan)
    flags = append_flag(flags, out["C_core"].isna(), "cannot_compute_C_core")

    clean_mask = (
        out["observed_debt_ratio"].between(0, 1, inclusive="neither")
        & out["tax_rate"].between(0, TAU_UPPER, inclusive="neither")
        & out["debt_cost"].between(0, R_UPPER, inclusive="neither")
        & out["C_core"].notna()
    )
    out["calibration_sample_flag"] = clean_mask
    flags = append_flag(flags, ~clean_mask, "excluded_from_robust_calibration")
    return out, flags, clean_mask


def calibrate_phi0(out: pd.DataFrame, clean_mask: pd.Series) -> dict[str, Any]:
    work = out[["firm_id", "period_date", "observed_debt_ratio", "C_core"]].copy()
    work["current_clean_mask"] = clean_mask.astype(bool)
    work = work.sort_values(["firm_id", "period_date"])
    work["next_observed_debt_ratio"] = work.groupby("firm_id")["observed_debt_ratio"].shift(-1)
    work["next_period_date"] = work.groupby("firm_id")["period_date"].shift(-1)
    work["delta_years"] = (work["next_period_date"] - work["period_date"]).dt.days / 365.25
    work["y"] = (work["next_observed_debt_ratio"] - work["observed_debt_ratio"]) / work["delta_years"]
    valid = (
        work["current_clean_mask"]
        & work["y"].notna()
        & work["C_core"].notna()
        & work["observed_debt_ratio"].notna()
        & work["next_observed_debt_ratio"].between(0, 1, inclusive="neither")
        & work["delta_years"].notna()
        & work["delta_years"].gt(0)
        & work["delta_years"].le(1.1)
        & np.isfinite(work["y"])
        & np.isfinite(work["C_core"])
        & np.isfinite(work["observed_debt_ratio"])
    )
    reg = work.loc[valid, ["y", "C_core", "observed_debt_ratio"]].copy()
    n_obs = int(len(reg))
    calibration: dict[str, Any] = {
        "variant": VARIANT,
        "label": LABEL,
        "allow_finance_expense_fallback": True,
        "force_finance_expense_proxy": True,
        "annual_only": False,
        "eta": ETA,
        "T": TARGET_HORIZON_YEARS,
        "n_obs": n_obs,
        "b_C": np.nan,
        "b_d": np.nan,
        "kappa_hat": np.nan,
        "s_hat": np.nan,
        "phi0_hat": np.nan,
        "calibration_valid": False,
        "reason_if_invalid": "",
        "calibration_sample_rows": int(valid.sum()),
    }
    if n_obs < MIN_CALIBRATION_OBS:
        calibration["reason_if_invalid"] = f"n_obs={n_obs} below min_calibration_obs={MIN_CALIBRATION_OBS}"
        return calibration
    coef, *_ = np.linalg.lstsq(
        reg[["C_core", "observed_debt_ratio"]].to_numpy(dtype=float),
        reg["y"].to_numpy(dtype=float),
        rcond=None,
    )
    b_c = float(coef[0])
    b_d = float(coef[1])
    calibration["b_C"] = b_c
    calibration["b_d"] = b_d
    if not np.isfinite(b_c) or not np.isfinite(b_d):
        calibration["reason_if_invalid"] = "non-finite dynamic adjustment coefficients"
    elif b_c <= 0:
        calibration["reason_if_invalid"] = f"b_C={b_c:.6g} <= 0"
    elif b_d >= 0:
        calibration["reason_if_invalid"] = f"b_d={b_d:.6g} >= 0"
    else:
        calibration["kappa_hat"] = -b_d
        calibration["s_hat"] = b_c / (-b_d)
        phi0_hat = ((-b_d) / b_c) ** (ETA - 1.0)
        calibration["phi0_hat"] = float(phi0_hat)
        calibration["calibration_valid"] = bool(np.isfinite(phi0_hat) and phi0_hat > 0)
        calibration["reason_if_invalid"] = "" if calibration["calibration_valid"] else f"invalid phi0={phi0_hat}"
    return calibration


def compute_optimal_gap(out: pd.DataFrame, calibration: dict[str, Any], flags: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    work = out.copy()
    work["phi0_hat"] = calibration.get("phi0_hat", np.nan)
    work["optimal_debt_ratio_raw"] = np.nan
    work["optimal_debt_ratio"] = np.nan
    work["is_optimal_debt_ratio_clipped"] = False
    work["leverage_gap_raw"] = np.nan
    work["leverage_gap"] = np.nan
    if calibration.get("calibration_valid"):
        a = 1.0 / (ETA - 1.0)
        numerator = work["tax_rate"] * ((1.0 + work["debt_cost"]) ** TARGET_HORIZON_YEARS - 1.0)
        denominator = float(calibration["phi0_hat"]) * TARGET_HORIZON_YEARS * ETA
        valid = (
            work["tax_rate"].notna()
            & work["debt_cost"].notna()
            & work["debt_cost"].gt(-1)
            & numerator.ge(0)
            & (denominator > 0)
        )
        raw_target = (numerator / denominator) ** a
        work["optimal_debt_ratio_raw"] = raw_target.where(valid)
        work["optimal_debt_ratio"] = work["optimal_debt_ratio_raw"].clip(lower=0.0, upper=DEBT_RATIO_UPPER)
        work["is_optimal_debt_ratio_clipped"] = work["optimal_debt_ratio_raw"].notna() & (
            work["optimal_debt_ratio_raw"].lt(0) | work["optimal_debt_ratio_raw"].gt(DEBT_RATIO_UPPER)
        )
        work["leverage_gap_raw"] = work["observed_debt_ratio"] - work["optimal_debt_ratio_raw"]
        work["leverage_gap"] = work["observed_debt_ratio"] - work["optimal_debt_ratio"]
        flags = append_flag(flags, work["optimal_debt_ratio_raw"].isna(), "cannot_compute_optimal_debt_ratio")
    else:
        flags = append_flag(flags, pd.Series(True, index=work.index), "phi0_calibration_invalid_no_optimal_debt_ratio")

    work["leverage_status"] = np.select(
        [
            work["leverage_gap"].isna(),
            work["leverage_gap"].abs().le(NEAR_OPTIMAL_TOL),
            work["leverage_gap"].gt(NEAR_OPTIMAL_TOL),
            work["leverage_gap"].lt(-NEAR_OPTIMAL_TOL),
        ],
        ["missing", "near_optimal", "over_levered", "under_levered"],
        default="missing",
    )
    return work, flags


def update_calibration_table(calibration: dict[str, Any]) -> None:
    path = OUTPUT_DIR / "phi0_calibration_robust_variants.csv"
    new_row = pd.DataFrame([calibration])
    if path.exists():
        current = pd.read_csv(path)
        current = current.loc[~current["variant"].eq(VARIANT)].copy()
        out = pd.concat([current, new_row], ignore_index=True, sort=False)
    else:
        out = new_row
    leading = [
        "variant",
        "label",
        "allow_finance_expense_fallback",
        "force_finance_expense_proxy",
        "eta",
        "T",
        "n_obs",
        "b_C",
        "b_d",
        "kappa_hat",
        "s_hat",
        "phi0_hat",
        "calibration_valid",
        "output_rows",
        "dstar_count",
        "dstar_median",
        "gap_median",
        "reason_if_invalid",
    ]
    out = out[[col for col in leading if col in out.columns] + [col for col in out.columns if col not in leading]]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def write_report(result: pd.DataFrame, calibration: dict[str, Any]) -> None:
    work = result.copy()
    work["period_date"] = pd.to_datetime(work["period_date"], errors="coerce")
    work["year"] = work["period_date"].dt.year
    year_counts = (
        work.groupby("year")
        .agg(
            rows=("firm_id", "size"),
            firms=("firm_id", "nunique"),
            finance_proxy_rows=("interest_source_flag", lambda s: s.eq("finance_expense_proxy").sum()),
            direct_rows=("interest_source_flag", lambda s: s.eq("direct_interest_expense").sum()),
        )
        .reset_index()
    )
    early = year_counts.loc[year_counts["year"].between(2015, 2018)]
    lines = [
        "# Consistent Finance-Expense Proxy Gap Variant",
        "",
        "## Scope",
        "",
        "- Variant: `D_finance_proxy_consistent`.",
        "- Interest input: always use finance expense `B001211000` as the interest-expense proxy when available.",
        "- Purpose: avoid a definition jump around 2018, when direct interest expense `B001211101` becomes broadly available.",
        "- Formula and clean filters match the existing robust variants.",
        "",
        "## Calibration",
        "",
        f"- n_obs: `{calibration.get('n_obs')}`.",
        f"- phi0_hat: `{calibration.get('phi0_hat')}`.",
        f"- calibration_valid: `{calibration.get('calibration_valid')}`.",
        "",
        "## 2015-2018 Coverage",
        "",
        early.to_markdown(index=False),
        "",
        "## Output",
        "",
        f"- `{OUTPUT_DIR / FILENAME}`",
        f"- `{OUTPUT_DIR / 'phi0_calibration_robust_variants.csv'}`",
    ]
    (OUTPUT_DIR / "variant_D_finance_proxy_consistent_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8-sig"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flags: pd.Series
    panel = build_panel()
    flags = pd.Series("", index=panel.index, dtype=object)
    panel, flags = compute_debt(panel, flags)
    panel, flags = compute_tax(panel, flags)
    panel, flags = compute_interest_and_cost(panel, flags)
    panel, flags, clean_mask = compute_core(panel, flags)
    calibration = calibrate_phi0(panel, clean_mask)
    panel, flags = compute_optimal_gap(panel, calibration, flags)

    panel["model_variant"] = VARIANT
    panel["variant_label"] = LABEL
    panel["data_quality_flags"] = flags.replace("", "ok")
    result = panel.loc[panel["calibration_sample_flag"]].copy()
    result_cols = [
        "model_variant",
        "firm_id",
        "period_date",
        "total_assets",
        "total_debt_used",
        "debt_source_flag",
        "observed_debt_ratio",
        "tax_rate_raw",
        "tax_rate",
        "interest_source_flag",
        "interest_expense_ytd_used",
        "period_interest_expense",
        "period_interest_years",
        "average_debt_for_cost",
        "debt_cost_raw",
        "debt_cost",
        "target_horizon_years",
        "eta",
        "C_core",
        "phi0_hat",
        "optimal_debt_ratio_raw",
        "optimal_debt_ratio",
        "leverage_gap_raw",
        "leverage_gap",
        "leverage_status",
        "is_optimal_debt_ratio_clipped",
        "calibration_sample_flag",
        "data_quality_flags",
    ]
    result = result[result_cols].sort_values(["firm_id", "period_date"]).reset_index(drop=True)
    calibration.update(
        {
            "output_rows": int(len(result)),
            "dstar_count": int(result["optimal_debt_ratio"].notna().sum()),
            "dstar_median": float(result["optimal_debt_ratio"].median()),
            "gap_median": float(result["leverage_gap"].median()),
        }
    )
    result.to_csv(OUTPUT_DIR / FILENAME, index=False, encoding="utf-8-sig")
    update_calibration_table(calibration)
    write_report(result, calibration)
    print(OUTPUT_DIR / FILENAME)


if __name__ == "__main__":
    main()
