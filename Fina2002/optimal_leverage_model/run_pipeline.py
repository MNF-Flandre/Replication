from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pq = None


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".dta"}
ZIP_EXTENSIONS = {".zip"}
TEXT_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(os.environ.get("QUANT_RAWDATA_ROOT", PROJECT_ROOT / "external_data")).expanduser()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

DEBT_RATIO_UPPER = 1.0
NEAR_OPTIMAL_TOL = 0.02
MIN_CALIBRATION_OBS = 30


EXACT_ALIASES: dict[str, list[str]] = {
    "firm_id": [
        "stkcd",
        "masterfundcode",
        "firmid",
        "stkcd",
        "stockcode",
        "secucode",
        "ticker",
        "symbol",
        "code",
        "公司代码",
        "股票代码",
        "证券代码",
        "基金主代码",
        "基金代码",
    ],
    "period_date": [
        "accper",
        "enddate",
        "reportdate",
        "date",
        "fiscalperiod",
        "period",
        "quarter",
        "statementdate",
        "截止日期",
        "报告期",
        "会计期间",
    ],
    "period_start_date": ["startdate", "start_date", "开始日期"],
    "report_type_id": ["reporttypeid", "reporttype", "datatypeid", "data_type_id", "定期报告类别编码"],
    "report_scope": ["typrep", "报表类型"],
    "total_assets": [
        "a001000000",
        "totalassets",
        "totalasset",
        "assetstotal",
        "total_asset",
        "assets_total",
        "at",
        "总资产",
        "资产总计",
    ],
    "net_assets": [
        "totaltna",
        "netassets",
        "netasset",
        "totalnetassets",
        "netass",
        "净资产",
        "净资产值netass",
        "基金资产净值",
    ],
    "total_debt": [
        "totaldebt",
        "interestbearingdebt",
        "interest_bearing_debt",
        "有息负债",
        "总债务",
    ],
    "total_liabilities": [
        "a002000000",
        "totalliabilities",
        "totalliability",
        "totalliab",
        "liabilities",
        "total_liabilities",
        "总负债",
        "负债合计",
    ],
    "short_term_debt": ["a002101000", "shorttermdebt", "short_term_debt", "短期借款", "短期债务"],
    "long_term_debt": ["a002201000", "longtermdebt", "long_term_debt", "长期借款", "长期债务"],
    "bonds_payable": ["a002203000", "bondspayable", "bonds_payable", "应付债券"],
    "notes_payable": ["a002107000", "notespayable", "notes_payable", "应付票据"],
    "current_portion_long_term_debt": [
        "a002125000",
        "currentportionlongtermdebt",
        "current_portion_long_term_debt",
        "一年内到期的非流动负债",
    ],
    "lease_liabilities": ["a002211000", "leaseliabilities", "lease_liabilities", "租赁负债"],
    "tax_rate": ["taxrate", "effectivetaxrate", "tax_rate", "effective_tax_rate", "实际税率"],
    "tax_expense": [
        "b002100000",
        "incometaxexpense",
        "taxexpense",
        "incometax",
        "income_tax",
        "income_tax_expense",
        "所得税费用",
    ],
    "pretax_income": [
        "b001000000",
        "pretaxincome",
        "incomebeforetax",
        "pretax_income",
        "income_before_tax",
        "利润总额",
        "税前利润",
    ],
    "debt_cost": ["debtcost", "interestrate", "borrowingcost", "debt_cost", "interest_rate", "borrowing_cost"],
    "interest_expense": ["b001211101", "bbd1102203", "interestexpense", "interest_expense", "利息费用"],
    "finance_expense": ["b001211000", "financeexpense", "finance_expense", "财务费用"],
}

CONTAINS_ALIASES: dict[str, list[str]] = {
    "firm_id": ["secu_code", "stock_code"],
    "period_date": ["enddt", "end_date"],
    "total_assets": ["totalasset", "总资产", "资产总计"],
    "net_assets": ["totaltna", "netass", "净资产"],
    "total_debt": ["totaldebt", "interestbearingdebt", "有息负债", "总债务"],
    "total_liabilities": ["totalliab", "liabilities", "总负债", "负债合计"],
    "tax_rate": ["taxrate", "effectivetaxrate"],
    "tax_expense": ["incometax", "taxexpense", "所得税"],
    "pretax_income": ["pretax", "beforetax", "利润总额", "税前"],
    "debt_cost": ["debtcost", "interestrate", "borrowingcost"],
    "interest_expense": ["interestexpense", "利息费用"],
    "finance_expense": ["financeexpense", "财务费用"],
}

DEBT_COMPONENT_FIELDS = [
    "short_term_debt",
    "long_term_debt",
    "bonds_payable",
    "notes_payable",
    "current_portion_long_term_debt",
    "lease_liabilities",
]

VALUE_FIELDS = [
    "total_assets",
    "net_assets",
    "total_debt",
    "total_liabilities",
    *DEBT_COMPONENT_FIELDS,
    "tax_rate",
    "tax_expense",
    "pretax_income",
    "debt_cost",
    "interest_expense",
    "finance_expense",
]


@dataclass(frozen=True)
class DataSource:
    path: Path
    extension: str
    size: int
    zip_member: str | None = None

    @property
    def ref(self) -> str:
        if self.zip_member:
            return f"{self.path}!{self.zip_member}"
        return str(self.path)

    @property
    def display_name(self) -> str:
        if self.zip_member:
            return f"{self.path.name}!{Path(self.zip_member).name}"
        return self.path.name


@dataclass
class SourceProfile:
    source: DataSource
    columns: list[str]
    reader: dict[str, Any]
    matches: dict[str, str]
    score: float
    read_error: str = ""


def normalize_name(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text)


def choose_column(columns: Iterable[str], field: str) -> str | None:
    cols = list(columns)
    normalized = {col: normalize_name(col) for col in cols}
    for alias in EXACT_ALIASES.get(field, []):
        needle = normalize_name(alias)
        for col, normed in normalized.items():
            if normed == needle:
                return col
    for alias in CONTAINS_ALIASES.get(field, []):
        needle = normalize_name(alias)
        if not needle:
            continue
        for col, normed in normalized.items():
            if needle in normed:
                return col
    return None


def all_matches(columns: Iterable[str]) -> dict[str, str]:
    fields = ["firm_id", "period_date", "period_start_date", "report_type_id", "report_scope", *VALUE_FIELDS]
    out: dict[str, str] = {}
    for field in fields:
        col = choose_column(columns, field)
        if col is not None:
            out[field] = col
    return out


def iter_sources(root: Path) -> list[DataSource]:
    sources: list[DataSource] = []
    skip_dirs = {".conda", "node_modules", "__pycache__", ".git"}

    def onerror(_error: OSError) -> None:
        return None

    for dirpath, dirnames, filenames in os.walk(root, onerror=onerror):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            path = Path(dirpath) / name
            suffix = path.suffix.lower()
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            if suffix in SUPPORTED_EXTENSIONS:
                sources.append(DataSource(path=path, extension=suffix, size=size))
            elif suffix in ZIP_EXTENSIONS:
                sources.extend(iter_zip_sources(path, size))
    return sources


def iter_zip_sources(path: Path, zip_size: int) -> list[DataSource]:
    out: list[DataSource] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                suffix = Path(info.filename).suffix.lower()
                if suffix in SUPPORTED_EXTENSIONS:
                    out.append(
                        DataSource(
                            path=path,
                            extension=suffix,
                            size=info.file_size or zip_size,
                            zip_member=info.filename,
                        )
                    )
    except Exception:
        return out
    return out


def read_columns(source: DataSource) -> tuple[list[str], dict[str, Any], str]:
    try:
        if source.zip_member:
            return read_zip_columns(source)
        return read_file_columns(source)
    except Exception as exc:
        return [], {}, f"{type(exc).__name__}: {exc}"


def read_zip_columns(source: DataSource) -> tuple[list[str], dict[str, Any], str]:
    assert source.zip_member is not None
    with zipfile.ZipFile(source.path) as zf:
        raw = zf.read(source.zip_member)
    return read_bytes_columns(raw, source.extension)


def read_file_columns(source: DataSource) -> tuple[list[str], dict[str, Any], str]:
    ext = source.extension
    if ext == ".csv":
        last_error = ""
        for encoding in TEXT_ENCODINGS:
            try:
                df = pd.read_csv(source.path, nrows=0, encoding=encoding, low_memory=False)
                return list(df.columns), {"kind": "csv", "encoding": encoding}, ""
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        return [], {}, last_error
    if ext in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(source.path)
        sheet_name = excel.sheet_names[0]
        df = pd.read_excel(source.path, sheet_name=sheet_name, nrows=0)
        return list(df.columns), {"kind": "excel", "sheet_name": sheet_name}, ""
    if ext == ".parquet":
        if pq is None:
            return [], {}, "pyarrow is not installed"
        parquet_file = pq.ParquetFile(source.path)
        return parquet_file.schema.names, {"kind": "parquet"}, ""
    if ext == ".feather":
        df = pd.read_feather(source.path, columns=[])
        return list(df.columns), {"kind": "feather"}, ""
    if ext == ".dta":
        reader = pd.read_stata(source.path, iterator=True)
        return list(reader.varlist), {"kind": "dta"}, ""
    return [], {}, f"unsupported extension {ext}"


def read_bytes_columns(raw: bytes, extension: str) -> tuple[list[str], dict[str, Any], str]:
    if extension == ".csv":
        last_error = ""
        for encoding in TEXT_ENCODINGS:
            try:
                df = pd.read_csv(io.BytesIO(raw), nrows=0, encoding=encoding, low_memory=False)
                return list(df.columns), {"kind": "zip_csv", "encoding": encoding}, ""
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        return [], {}, last_error
    if extension in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(io.BytesIO(raw))
        sheet_name = excel.sheet_names[0]
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, nrows=0)
        return list(df.columns), {"kind": "zip_excel", "sheet_name": sheet_name}, ""
    if extension == ".parquet":
        if pq is None:
            return [], {}, "pyarrow is not installed"
        parquet_file = pq.ParquetFile(io.BytesIO(raw))
        return parquet_file.schema.names, {"kind": "zip_parquet"}, ""
    if extension == ".feather":
        df = pd.read_feather(io.BytesIO(raw), columns=[])
        return list(df.columns), {"kind": "zip_feather"}, ""
    if extension == ".dta":
        df = pd.read_stata(io.BytesIO(raw), iterator=True)
        return list(df.varlist), {"kind": "zip_dta"}, ""
    return [], {}, f"unsupported extension {extension}"


def profile_score(source: DataSource, matches: dict[str, str]) -> float:
    score = 0.0
    if "firm_id" in matches:
        score += 5
    if "period_date" in matches:
        score += 5
    weights = {
        "total_assets": 12,
        "net_assets": 7,
        "total_debt": 15,
        "total_liabilities": 12,
        "tax_rate": 10,
        "tax_expense": 6,
        "pretax_income": 8,
        "debt_cost": 10,
        "interest_expense": 8,
        "finance_expense": 5,
    }
    for field, weight in weights.items():
        if field in matches:
            score += weight
    for field in DEBT_COMPONENT_FIELDS:
        if field in matches:
            score += 4
    path_text = source.ref.lower()
    if "\\raw data\\" in path_text or "/raw data/" in path_text or "\\rawdata\\" in path_text or "/rawdata/" in path_text:
        score += 8
    if "\\raw_package\\" in path_text or "/raw_package/" in path_text:
        score += 4
    if any(token in path_text for token in ["balance", "资产负债", "fund_fin_balance", "fs_combas"]):
        score += 8
    if any(token in path_text for token in ["income", "业绩与收益", "fund_fin_income", "fs_comins"]):
        score += 8
    if any(token in path_text for token in ["fin_index", "财务指标"]):
        score += 5
    if any(token in path_text for token in ["baseline", "date_runs", "node_modules", "\\output\\", "/output/"]):
        score -= 15
    if source.zip_member:
        score -= 2
    return score


def scan_sources(root: Path) -> list[SourceProfile]:
    profiles: list[SourceProfile] = []
    for source in iter_sources(root):
        columns, reader, error = read_columns(source)
        matches = all_matches(columns)
        score = profile_score(source, matches) if matches else 0.0
        profiles.append(SourceProfile(source=source, columns=columns, reader=reader, matches=matches, score=score, read_error=error))
    return profiles


def field_source_score(profile: SourceProfile, field: str) -> float:
    if field not in profile.matches:
        return -math.inf
    if "firm_id" not in profile.matches or "period_date" not in profile.matches:
        return -math.inf
    score = profile.score
    text = profile.source.ref.lower()
    column = normalize_name(profile.matches[field])
    if field in {"total_assets", "total_debt", "total_liabilities", *DEBT_COMPONENT_FIELDS}:
        if any(token in text for token in ["balance", "资产负债", "fund_fin_balance", "fs_combas"]):
            score += 50
        if any(token in text for token in ["flow", "allocation", "portfolio"]):
            score -= 20
    if field == "net_assets":
        if any(token in text for token in ["fin_index", "财务指标"]):
            score += 50
        if "totaltna" in column or "netass" in column:
            score += 15
    if field in {"tax_rate", "tax_expense", "pretax_income", "debt_cost", "interest_expense", "finance_expense"}:
        if any(token in text for token in ["income", "业绩与收益", "fund_fin_income", "fs_comins"]):
            score += 50
        if field == "finance_expense" and "financeexpense" in column:
            score += 10
        if field == "tax_expense" and "incometax" in column:
            score += 10
    if "\\merged data\\" in text or "/merged data/" in text:
        score -= 5
    if "\\descriptive\\" in text or "/descriptive/" in text:
        score -= 5
    if profile.source.zip_member:
        score -= 3
    return score


def select_profiles(profiles: list[SourceProfile]) -> tuple[list[SourceProfile], dict[str, dict[str, Any]]]:
    selected_refs: set[str] = set()
    selected: list[SourceProfile] = []
    mapping: dict[str, dict[str, Any]] = {}

    fields_to_select = [
        "total_assets",
        "net_assets",
        "total_debt",
        "total_liabilities",
        *DEBT_COMPONENT_FIELDS,
        "tax_rate",
        "tax_expense",
        "pretax_income",
        "debt_cost",
        "interest_expense",
        "finance_expense",
    ]
    for field in fields_to_select:
        candidates = [(field_source_score(profile, field), profile) for profile in profiles]
        candidates = [(score, profile) for score, profile in candidates if math.isfinite(score)]
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], -item[1].source.size, item[1].source.ref))
        score, profile = candidates[0]
        mapping[field] = {
            "source": profile.source.ref,
            "column": profile.matches[field],
            "source_score": round(score, 3),
        }
        if profile.source.ref not in selected_refs:
            selected_refs.add(profile.source.ref)
            selected.append(profile)

    selected.sort(key=lambda p: (-p.score, p.source.ref))
    return selected, mapping


def read_profile_data(profile: SourceProfile, selected_mapping: dict[str, dict[str, Any]]) -> pd.DataFrame:
    needed: set[str] = set()
    for key in ["firm_id", "period_date", "period_start_date", "report_type_id", "report_scope"]:
        if key in profile.matches:
            needed.add(profile.matches[key])
    for field, item in selected_mapping.items():
        if item["source"] == profile.source.ref and field in profile.matches:
            needed.add(profile.matches[field])
    if not needed:
        return pd.DataFrame()

    usecols = [col for col in profile.columns if col in needed]
    df = read_source_dataframe(profile, usecols)
    rename: dict[str, str] = {}
    for canonical, source_col in profile.matches.items():
        if source_col in usecols:
            rename[source_col] = canonical
    df = df.rename(columns=rename)
    keep_cols = [col for col in ["firm_id", "period_date", "period_start_date", "report_type_id", "report_scope", *VALUE_FIELDS] if col in df.columns]
    df = df[keep_cols].copy()
    df["__source_ref"] = profile.source.ref
    return df


def read_source_dataframe(profile: SourceProfile, usecols: list[str]) -> pd.DataFrame:
    source = profile.source
    reader = profile.reader
    kind = reader.get("kind", "")
    if source.zip_member:
        with zipfile.ZipFile(source.path) as zf:
            raw = zf.read(source.zip_member)
        buffer = io.BytesIO(raw)
        if kind == "zip_csv":
            return pd.read_csv(buffer, usecols=usecols, encoding=reader["encoding"], low_memory=False)
        if kind == "zip_excel":
            return pd.read_excel(buffer, sheet_name=reader["sheet_name"], usecols=usecols)
        if kind == "zip_parquet":
            return pd.read_parquet(buffer, columns=usecols)
        if kind == "zip_feather":
            return pd.read_feather(buffer, columns=usecols)
        if kind == "zip_dta":
            return pd.read_stata(buffer, columns=usecols)
    if kind == "csv":
        return pd.read_csv(source.path, usecols=usecols, encoding=reader["encoding"], low_memory=False)
    if kind == "excel":
        return pd.read_excel(source.path, sheet_name=reader["sheet_name"], usecols=usecols)
    if kind == "parquet":
        return pd.read_parquet(source.path, columns=usecols)
    if kind == "feather":
        return pd.read_feather(source.path, columns=usecols)
    if kind == "dta":
        return pd.read_stata(source.path, columns=usecols)
    raise ValueError(f"Unsupported reader kind for {source.ref}: {kind}")


def normalize_firm_id(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def first_non_null(series: pd.Series) -> Any:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def normalize_panel_source(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    out = df.copy()
    out["firm_id"] = out["firm_id"].map(normalize_firm_id) if "firm_id" in out.columns else np.nan
    out["period_date"] = pd.to_datetime(out["period_date"], errors="coerce") if "period_date" in out.columns else pd.NaT
    if "period_start_date" in out.columns:
        out["period_start_date"] = pd.to_datetime(out["period_start_date"], errors="coerce")
    else:
        out["period_start_date"] = pd.NaT
    if "report_type_id" in out.columns:
        out["report_type_id"] = out["report_type_id"].map(normalize_firm_id)
    else:
        out["report_type_id"] = np.nan
    if "report_scope" in out.columns:
        out["report_scope"] = out["report_scope"].astype(str).str.strip().str.upper()
        if out["report_scope"].eq("A").any():
            out = out[out["report_scope"].eq("A")].copy()
    else:
        out["report_scope"] = np.nan
    out["period_key"] = out["report_type_id"].fillna("")
    out = out[out["firm_id"].notna() & out["period_date"].notna()].copy()
    for col in VALUE_FIELDS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    key_cols = ["firm_id", "period_date", "period_key"]
    duplicate_count = int(out.duplicated(key_cols, keep=False).sum())
    aggregations: dict[str, Any] = {}
    for col in out.columns:
        if col in key_cols:
            continue
        aggregations[col] = first_non_null
    out = out.groupby(key_cols, as_index=False, dropna=False).agg(aggregations)
    return out, duplicate_count


def build_panel(selected_profiles: list[SourceProfile], selected_mapping: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_frames: list[pd.DataFrame] = []
    duplicate_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}

    for profile in selected_profiles:
        raw = read_profile_data(profile, selected_mapping)
        row_counts[profile.source.ref] = int(len(raw))
        normalized, duplicate_count = normalize_panel_source(raw)
        duplicate_counts[profile.source.ref] = duplicate_count
        if not normalized.empty:
            source_frames.append(normalized)

    if not source_frames:
        return pd.DataFrame(), {"source_duplicate_rows": duplicate_counts, "source_row_counts": row_counts}

    panel = source_frames[0]
    for frame in source_frames[1:]:
        panel = panel.merge(frame, on=["firm_id", "period_date", "period_key"], how="outer", suffixes=("", "__new"))
        for col in list(panel.columns):
            if not col.endswith("__new"):
                continue
            base_col = col[:-5]
            if base_col in panel.columns:
                panel[base_col] = panel[base_col].combine_first(panel[col])
                panel = panel.drop(columns=[col])
            else:
                panel = panel.rename(columns={col: base_col})
    return panel, {"source_duplicate_rows": duplicate_counts, "source_row_counts": row_counts}


def append_flag(flags: pd.Series, mask: pd.Series | np.ndarray, flag: str) -> pd.Series:
    mask = pd.Series(mask, index=flags.index).fillna(False)
    flags.loc[mask] = np.where(flags.loc[mask].eq(""), flag, flags.loc[mask] + ";" + flag)
    return flags


def infer_period_type_from_report_type(report_type: Any) -> str:
    if pd.isna(report_type):
        return "unknown"
    text = str(report_type).strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    if text in {"1", "2", "3", "4", "q1", "q2", "q3", "q4", "quarter", "quarterly"}:
        return "quarterly"
    if text in {"5", "h1", "h2", "half", "semiannual", "半年度", "中报", "半年报"}:
        return "semiannual"
    if text in {"6", "annual", "year", "yearly", "年度", "年报"}:
        return "annual"
    return "other"


def infer_period_type_from_date(period_date: Any) -> str:
    if pd.isna(period_date):
        return "unknown"
    ts = pd.Timestamp(period_date)
    month_day = (ts.month, ts.day)
    if month_day in {(3, 31), (9, 30)}:
        return "quarterly"
    if month_day == (6, 30):
        return "semiannual"
    if month_day == (12, 31):
        return "annual"
    return "unknown"


def cumulative_period_years(report_type: Any, start_date: Any, end_date: Any, period_type: str) -> float:
    if not pd.isna(report_type):
        text = str(report_type).strip().lower()
        if text.endswith(".0"):
            text = text[:-2]
        mapping = {
            "1": 0.25,
            "2": 0.50,
            "3": 0.75,
            "4": 1.00,
            "5": 0.50,
            "6": 1.00,
            "q1": 0.25,
            "q2": 0.50,
            "q3": 0.75,
            "q4": 1.00,
        }
        if text in mapping:
            return mapping[text]
    if not pd.isna(start_date) and not pd.isna(end_date):
        days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        if days > 0:
            return days / 365.25
    if not pd.isna(end_date):
        month_day = (pd.Timestamp(end_date).month, pd.Timestamp(end_date).day)
        coverage = {
            (3, 31): 0.25,
            (6, 30): 0.50,
            (9, 30): 0.75,
            (12, 31): 1.00,
        }
        if month_day in coverage:
            return coverage[month_day]
    if period_type == "quarterly":
        return 0.25
    if period_type == "semiannual":
        return 0.50
    if period_type == "annual":
        return 1.00
    return np.nan


def complete_period_types(panel: pd.DataFrame) -> pd.Series:
    period_type = panel.get("report_type_id", pd.Series(index=panel.index, dtype=object)).map(infer_period_type_from_report_type)
    date_inferred = panel.get("period_date", pd.Series(index=panel.index, dtype="datetime64[ns]")).map(infer_period_type_from_date)
    period_type = period_type.where(~period_type.isin(["unknown", "other"]), date_inferred)
    unknown = period_type.eq("unknown")
    if unknown.any():
        sorted_panel = panel.sort_values(["firm_id", "period_date"])
        deltas = sorted_panel.groupby("firm_id")["period_date"].diff().dt.days
        inferred = pd.Series(index=sorted_panel.index, dtype=object)
        inferred.loc[deltas.between(70, 120)] = "quarterly"
        inferred.loc[deltas.between(150, 220)] = "semiannual"
        inferred.loc[deltas.between(300, 400)] = "annual"
        period_type.loc[inferred.index] = period_type.loc[inferred.index].where(~period_type.loc[inferred.index].eq("unknown"), inferred)
    return period_type.fillna("unknown")


def compute_model(panel: pd.DataFrame, eta: float, target_horizon_years: float, min_calibration_obs: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if eta <= 1:
        raise ValueError("eta must be greater than 1")
    if target_horizon_years <= 0:
        raise ValueError("target_horizon_years must be positive")

    out = panel.copy()
    flags = pd.Series("", index=out.index, dtype=object)
    out["period_type"] = complete_period_types(out) if not out.empty else pd.Series(dtype=object)
    out["period_coverage_years"] = [
        cumulative_period_years(rt, st, ed, pt)
        for rt, st, ed, pt in zip(
            out.get("report_type_id", pd.Series(index=out.index, dtype=object)),
            out.get("period_start_date", pd.Series(index=out.index, dtype="datetime64[ns]")),
            out.get("period_date", pd.Series(index=out.index, dtype="datetime64[ns]")),
            out.get("period_type", pd.Series(index=out.index, dtype=object)),
        )
    ]

    for col in ["total_assets", "net_assets", "total_debt", "total_liabilities", "tax_rate", "tax_expense", "pretax_income", "debt_cost", "interest_expense", "finance_expense"]:
        if col not in out.columns:
            out[col] = np.nan

    out["total_debt_used"] = np.nan
    out["debt_source_flag"] = ""

    if out["total_debt"].notna().any():
        out["total_debt_used"] = out["total_debt"]
        out["debt_source_flag"] = "direct_total_debt_or_interest_bearing_debt"
    component_cols = [col for col in DEBT_COMPONENT_FIELDS if col in out.columns and out[col].notna().any()]
    if component_cols:
        component_debt = out[component_cols].sum(axis=1, min_count=1)
        fill_mask = out["total_debt_used"].isna() & component_debt.notna()
        out.loc[fill_mask, "total_debt_used"] = component_debt.loc[fill_mask]
        out.loc[fill_mask, "debt_source_flag"] = "sum_interest_bearing_debt_components"
    if out["total_liabilities"].notna().any():
        fill_mask = out["total_debt_used"].isna() & out["total_liabilities"].notna()
        out.loc[fill_mask, "total_debt_used"] = out.loc[fill_mask, "total_liabilities"]
        out.loc[fill_mask, "debt_source_flag"] = "total_liabilities_proxy"
        flags = append_flag(flags, fill_mask, "observed_leverage_used_total_liabilities_proxy")
    if out["total_debt_used"].isna().any() and out["total_assets"].notna().any() and out["net_assets"].notna().any():
        derived_liabilities = out["total_assets"] - out["net_assets"]
        negative = derived_liabilities < -1e-8 * out["total_assets"].abs().fillna(0)
        fill_mask = out["total_debt_used"].isna() & derived_liabilities.notna() & ~negative
        flags = append_flag(flags, fill_mask, "observed_leverage_used_total_assets_minus_net_assets_proxy")
        flags = append_flag(flags, out["total_debt_used"].isna() & negative, "negative_derived_liabilities")
        derived_liabilities = derived_liabilities.mask(derived_liabilities.abs() < 1e-8, 0.0)
        out.loc[fill_mask, "total_debt_used"] = derived_liabilities.loc[fill_mask]
        out.loc[fill_mask, "debt_source_flag"] = "total_liabilities_proxy_from_total_assets_minus_net_assets"

    valid_observed = (out["total_assets"] > 0) & (out["total_debt_used"] >= 0)
    out["observed_debt_ratio"] = np.where(valid_observed, out["total_debt_used"] / out["total_assets"], np.nan)
    flags = append_flag(flags, out["total_assets"].isna(), "missing_total_assets")
    flags = append_flag(flags, out["total_assets"].notna() & (out["total_assets"] <= 0), "non_positive_total_assets")
    flags = append_flag(flags, out["total_debt_used"].isna(), "missing_debt_measure")
    flags = append_flag(flags, out["observed_debt_ratio"].notna() & (out["observed_debt_ratio"] > 1), "observed_debt_ratio_above_1")

    out["tax_rate_raw"] = np.nan
    if out["tax_rate"].notna().any():
        raw = out["tax_rate"].astype(float)
        finite = raw[np.isfinite(raw)]
        if not finite.empty and finite.abs().median() > 1 and finite.abs().median() <= 100:
            raw = raw / 100.0
            flags = append_flag(flags, out["tax_rate"].notna(), "tax_rate_interpreted_as_percent")
        out["tax_rate_raw"] = raw
    elif out["tax_expense"].notna().any() and out["pretax_income"].notna().any():
        raw = out["tax_expense"] / out["pretax_income"]
        non_positive_pretax = out["pretax_income"].notna() & (out["pretax_income"] <= 0)
        raw = raw.mask(non_positive_pretax, 0.0)
        out["tax_rate_raw"] = raw
        flags = append_flag(flags, non_positive_pretax, "pretax_income_non_positive_tax_rate_set_to_0")
    else:
        flags = append_flag(flags, pd.Series(True, index=out.index), "missing_tax_rate_inputs")
    out["tax_rate"] = out["tax_rate_raw"].clip(lower=0.0, upper=1.0)
    flags = append_flag(flags, out["tax_rate_raw"].notna() & (out["tax_rate_raw"] != out["tax_rate"]), "tax_rate_clipped_to_0_1")
    flags = append_flag(flags, out["tax_rate"].isna(), "cannot_compute_tax_rate")

    out["__data_quality_flags_work"] = flags
    out = out.sort_values(["firm_id", "period_date", "period_key"]).reset_index(drop=True)
    flags = out.pop("__data_quality_flags_work").reset_index(drop=True)

    direct_debt_cost_input = out["debt_cost"].copy()
    out["debt_cost_raw"] = np.nan
    out["debt_cost"] = np.nan
    if direct_debt_cost_input.notna().any():
        direct = direct_debt_cost_input.astype(float)
        finite = direct[np.isfinite(direct)]
        if not finite.empty and finite.abs().median() > 1 and finite.abs().median() <= 100:
            direct = direct / 100.0
            flags = append_flag(flags, direct_debt_cost_input.notna(), "debt_cost_interpreted_as_percent")
        out["debt_cost_raw"] = direct
        out["debt_cost"] = direct.where(direct > -1)
    else:
        interest_source = None
        if out["interest_expense"].notna().any():
            interest_source = "interest_expense"
            interest = out["interest_expense"].copy()
            finance_fill_mask = interest.isna() & out["finance_expense"].notna()
            if finance_fill_mask.any():
                interest.loc[finance_fill_mask] = out.loc[finance_fill_mask, "finance_expense"]
                flags = append_flag(flags, finance_fill_mask, "finance_expense_used_as_interest_expense_proxy")
        elif out["finance_expense"].notna().any():
            interest_source = "finance_expense"
            interest = out["finance_expense"]
            flags = append_flag(flags, out["finance_expense"].notna(), "finance_expense_used_as_interest_expense_proxy")
        else:
            interest = pd.Series(np.nan, index=out.index)

        lagged_debt = out.groupby("firm_id")["total_debt_used"].shift(1)
        period_years = pd.to_numeric(out["period_coverage_years"], errors="coerce")
        raw_period_cost = interest / lagged_debt
        valid_cost = (lagged_debt > 0) & (period_years > 0) & raw_period_cost.notna() & (raw_period_cost > -1)
        out["debt_cost_raw"] = raw_period_cost.where(valid_cost)
        out["debt_cost"] = ((1.0 + raw_period_cost) ** (1.0 / period_years) - 1.0).where(valid_cost)
        if interest_source is None:
            flags = append_flag(flags, pd.Series(True, index=out.index), "missing_interest_expense")
        else:
            flags = append_flag(flags, interest.isna(), "missing_interest_expense")
        flags = append_flag(flags, lagged_debt.isna() | (lagged_debt <= 0), "missing_or_non_positive_lagged_debt_for_debt_cost")
        flags = append_flag(flags, period_years.isna() | (period_years <= 0), "missing_period_length_for_debt_cost")
        flags = append_flag(flags, raw_period_cost.notna() & (raw_period_cost <= -1), "invalid_period_debt_cost_le_minus_1")
        if out["debt_source_flag"].str.contains("proxy", na=False).any():
            flags = append_flag(flags, out["debt_cost"].notna(), "debt_cost_uses_lagged_proxy_debt")

    flags = append_flag(flags, out["debt_cost"].isna(), "cannot_compute_debt_cost")
    flags = append_flag(flags, out["debt_cost"].notna() & (out["debt_cost"] < -1), "annualized_debt_cost_below_minus_1")
    flags = append_flag(flags, out["debt_cost"].notna() & (out["debt_cost"] > 1), "annualized_debt_cost_above_100pct")

    a = 1.0 / (eta - 1.0)
    out["target_horizon_years"] = target_horizon_years
    out["eta"] = eta
    numerator_core = out["tax_rate"] * ((1.0 + out["debt_cost"]) ** target_horizon_years - 1.0)
    denominator_core = target_horizon_years * eta
    valid_core = out["tax_rate"].notna() & out["debt_cost"].notna() & (out["debt_cost"] > -1) & (numerator_core >= 0) & np.isfinite(numerator_core)
    out["C_core"] = np.where(valid_core, (numerator_core / denominator_core) ** a, np.nan)
    flags = append_flag(flags, out["C_core"].isna(), "cannot_compute_C_core")

    calibration = calibrate_phi0(out, eta=eta, min_calibration_obs=min_calibration_obs)
    phi0_hat = calibration.get("phi0_hat", np.nan)
    out["phi0_hat"] = phi0_hat
    out["optimal_debt_ratio_raw"] = np.nan
    out["optimal_debt_ratio"] = np.nan
    out["is_optimal_debt_ratio_clipped"] = False
    out["leverage_gap_raw"] = np.nan
    out["leverage_gap"] = np.nan
    if calibration["calibration_valid"]:
        numerator = out["tax_rate"] * ((1.0 + out["debt_cost"]) ** target_horizon_years - 1.0)
        denominator = phi0_hat * target_horizon_years * eta
        valid_target = out["tax_rate"].notna() & out["debt_cost"].notna() & (out["debt_cost"] > -1) & (numerator >= 0) & (denominator > 0)
        raw_target = (numerator / denominator) ** a
        out["optimal_debt_ratio_raw"] = raw_target.where(valid_target)
        out["optimal_debt_ratio"] = out["optimal_debt_ratio_raw"].clip(lower=0.0, upper=DEBT_RATIO_UPPER)
        out["is_optimal_debt_ratio_clipped"] = out["optimal_debt_ratio_raw"].notna() & (
            (out["optimal_debt_ratio_raw"] < 0) | (out["optimal_debt_ratio_raw"] > DEBT_RATIO_UPPER)
        )
        out["leverage_gap_raw"] = out["observed_debt_ratio"] - out["optimal_debt_ratio_raw"]
        out["leverage_gap"] = out["observed_debt_ratio"] - out["optimal_debt_ratio"]
        flags = append_flag(flags, out["optimal_debt_ratio_raw"].isna(), "cannot_compute_optimal_debt_ratio")
        flags = append_flag(flags, out["is_optimal_debt_ratio_clipped"], "optimal_debt_ratio_clipped")
    else:
        flags = append_flag(flags, pd.Series(True, index=out.index), "phi0_calibration_invalid_no_optimal_debt_ratio")

    out["leverage_status"] = np.select(
        [
            out["leverage_gap"].isna(),
            out["leverage_gap"].abs() <= NEAR_OPTIMAL_TOL,
            out["leverage_gap"] > NEAR_OPTIMAL_TOL,
            out["leverage_gap"] < -NEAR_OPTIMAL_TOL,
        ],
        ["missing", "near_optimal", "over_levered", "under_levered"],
        default="missing",
    )
    out["data_quality_flags"] = flags.replace("", "ok")

    result_cols = [
        "firm_id",
        "period_date",
        "period_type",
        "total_assets",
        "total_debt_used",
        "debt_source_flag",
        "observed_debt_ratio",
        "tax_rate_raw",
        "tax_rate",
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
        "data_quality_flags",
    ]
    for col in result_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[result_cols], calibration


def calibrate_phi0(out: pd.DataFrame, eta: float, min_calibration_obs: int) -> dict[str, Any]:
    work = out[["firm_id", "period_date", "observed_debt_ratio", "C_core"]].copy()
    work = work.sort_values(["firm_id", "period_date"])
    work["next_observed_debt_ratio"] = work.groupby("firm_id")["observed_debt_ratio"].shift(-1)
    work["next_period_date"] = work.groupby("firm_id")["period_date"].shift(-1)
    work["delta_years"] = (work["next_period_date"] - work["period_date"]).dt.days / 365.25
    work["y"] = (work["next_observed_debt_ratio"] - work["observed_debt_ratio"]) / work["delta_years"]
    valid = (
        work["y"].notna()
        & work["C_core"].notna()
        & work["observed_debt_ratio"].notna()
        & work["delta_years"].notna()
        & (work["delta_years"] > 0)
        & np.isfinite(work["y"])
        & np.isfinite(work["C_core"])
        & np.isfinite(work["observed_debt_ratio"])
    )
    reg = work.loc[valid, ["y", "C_core", "observed_debt_ratio"]].copy()
    n_obs = int(len(reg))
    reason = ""
    b_c = np.nan
    b_d = np.nan
    kappa_hat = np.nan
    s_hat = np.nan
    phi0_hat = np.nan
    calibration_valid = False

    if n_obs < min_calibration_obs:
        reason = f"可用于动态调整校准的样本数 {n_obs} 少于 min_calibration_obs={min_calibration_obs}。"
    else:
        x = reg[["C_core", "observed_debt_ratio"]].to_numpy(dtype=float)
        y = reg["y"].to_numpy(dtype=float)
        try:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            b_c = float(coef[0])
            b_d = float(coef[1])
            if not np.isfinite(b_c) or not np.isfinite(b_d):
                reason = "无截距回归系数不是有限数。"
            elif b_c <= 0:
                reason = f"动态调整回归得到 b_C={b_c:.6g} <= 0。"
            elif b_d >= 0:
                reason = f"动态调整回归得到 b_d={b_d:.6g} >= 0。"
            else:
                kappa_hat = -b_d
                s_hat = b_c / (-b_d)
                phi0_hat = ((-b_d) / b_c) ** (eta - 1.0)
                if not np.isfinite(phi0_hat) or phi0_hat <= 0:
                    reason = f"估计出的 phi0={phi0_hat} <= 0 或非有限。"
                else:
                    calibration_valid = True
                    reason = "ok"
        except Exception as exc:
            reason = f"无截距回归失败：{type(exc).__name__}: {exc}"

    return {
        "eta": eta,
        "T": float(out["target_horizon_years"].iloc[0]) if "target_horizon_years" in out.columns and len(out) else np.nan,
        "n_obs": n_obs,
        "b_C": b_c,
        "b_d": b_d,
        "kappa_hat": kappa_hat,
        "s_hat": s_hat,
        "phi0_hat": phi0_hat,
        "calibration_valid": bool(calibration_valid),
        "reason_if_invalid": "" if calibration_valid else reason,
        "calibration_sample_rows": int(valid.sum()),
    }


def percent_nonzero_zero(series: pd.Series) -> float:
    denom = series.notna().sum()
    if denom == 0:
        return np.nan
    return float((series.fillna(np.nan) == 0).sum() / denom)


def distribution_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        desc = {
            "variable": col,
            "count": int(series.notna().sum()),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "p1": series.quantile(0.01),
            "p5": series.quantile(0.05),
            "p25": series.quantile(0.25),
            "median": series.quantile(0.50),
            "p75": series.quantile(0.75),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99),
            "max": series.max(),
        }
        rows.append(desc)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, float_digits: int = 6) -> str:
    if df.empty:
        return "(empty)"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}g}")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(work.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in headers) + " |")
    return "\n".join(lines)


def write_manifest(profiles: list[SourceProfile], path: Path) -> None:
    rows = []
    for profile in profiles:
        rows.append(
            {
                "source": profile.source.ref,
                "extension": profile.source.extension,
                "size": profile.source.size,
                "score": profile.score,
                "columns": json.dumps(profile.columns, ensure_ascii=False),
                "matches": json.dumps(profile.matches, ensure_ascii=False),
                "read_error": profile.read_error,
            }
        )
    pd.DataFrame(rows).sort_values(["score", "size"], ascending=[False, False]).to_csv(path, index=False, encoding="utf-8-sig")


def write_field_mapping(mapping: dict[str, dict[str, Any]], selected_profiles: list[SourceProfile], path: Path) -> None:
    payload = {
        "selected_field_mapping": mapping,
        "selected_sources": [
            {
                "source": profile.source.ref,
                "columns": profile.columns,
                "matches": profile.matches,
                "reader": profile.reader,
                "score": profile.score,
            }
            for profile in selected_profiles
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_calibration_outputs(calibration: dict[str, Any], output_dir: Path) -> None:
    pd.DataFrame([calibration]).to_csv(output_dir / "phi0_calibration.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# phi0 校准结果",
        "",
        markdown_table(pd.DataFrame([calibration])),
        "",
    ]
    if calibration["calibration_valid"]:
        lines.append("校准有效，后续 d* 使用该 phi0_hat 计算。")
    else:
        lines.append(f"模型校准失败：{calibration['reason_if_invalid']}")
        lines.append("")
        lines.append("因此本轮不会强行计算 phi0，也不会编造 d* 或 Gap。")
    (output_dir / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_missing_report(
    result: pd.DataFrame,
    mapping: dict[str, dict[str, Any]],
    selected_profiles: list[SourceProfile],
    build_info: dict[str, Any],
    calibration: dict[str, Any],
    output_dir: Path,
) -> None:
    required = {
        "firm_id": "企业标识，用于构造企业-时间面板并计算下一期负债率变化。",
        "period_date": "时间字段，用于排序、识别季度/半年度和计算 Delta years。",
        "total_assets": "K_it，实际负债率 d_obs = D/K 的分母。",
        "debt_measure": "D_it，优先有息债务；若无则总负债 proxy。",
        "tax_rate_or_tax_and_pretax": "tau_it，d* 分子中的有效税率。",
        "debt_cost_or_interest_and_lagged_debt": "r_it，d* 分子中的年化债务成本。",
    }
    present = {
        "firm_id": any("firm_id" in p.matches for p in selected_profiles),
        "period_date": any("period_date" in p.matches for p in selected_profiles),
        "total_assets": "total_assets" in mapping,
        "debt_measure": any(field in mapping for field in ["total_debt", "total_liabilities", *DEBT_COMPONENT_FIELDS])
        or ("total_assets" in mapping and "net_assets" in mapping),
        "tax_rate_or_tax_and_pretax": "tax_rate" in mapping or ("tax_expense" in mapping and "pretax_income" in mapping),
        "debt_cost_or_interest_and_lagged_debt": "debt_cost" in mapping or any(field in mapping for field in ["interest_expense", "finance_expense"]),
    }
    missing_rows = [
        {"required_item": key, "purpose": purpose}
        for key, purpose in required.items()
        if not present.get(key, False)
    ]
    fallback_lines = []
    if "total_debt" not in mapping and any(field in mapping for field in DEBT_COMPONENT_FIELDS):
        used_components = ", ".join(field for field in DEBT_COMPONENT_FIELDS if field in mapping)
        fallback_lines.append(f"- 债务规模优先使用有息债务构成项加总：{used_components}。")
    if "total_debt" not in mapping and "total_liabilities" in mapping:
        if any(field in mapping for field in DEBT_COMPONENT_FIELDS):
            fallback_lines.append("- 对无法由有息债务构成项计算 D_it 的行，使用 total liabilities / 总负债 proxy。")
        else:
            fallback_lines.append("- 债务规模使用 total liabilities / 总负债 proxy。")
    if "total_debt" not in mapping and "total_liabilities" not in mapping and "total_assets" in mapping and "net_assets" in mapping:
        fallback_lines.append("- 债务规模使用 total_assets - net_assets 推导出的总负债 proxy；observed leverage used total liabilities as proxy。")
    if "interest_expense" not in mapping and "finance_expense" in mapping:
        fallback_lines.append("- 债务成本中的利息支出使用 finance expense / 财务费用 proxy，已在 data_quality_flags 标记。")

    counts = {
        "无法计算 tax_rate 的样本数": int(result["tax_rate"].isna().sum()) if "tax_rate" in result else 0,
        "无法计算 debt_cost 的样本数": int(result["debt_cost"].isna().sum()) if "debt_cost" in result else 0,
        "无法计算 observed_debt_ratio 的样本数": int(result["observed_debt_ratio"].isna().sum()) if "observed_debt_ratio" in result else 0,
        "无法进入 calibration 的样本数": int(len(result) - calibration.get("n_obs", 0)),
        "同一源文件内重复 firm-period 样本数": int(sum(build_info.get("source_duplicate_rows", {}).values())),
    }
    lines = [
        "# 缺失数据报告",
        "",
        "## 必要字段缺失",
        "",
        markdown_table(pd.DataFrame(missing_rows)) if missing_rows else "未发现结构性必需字段完全缺失。",
        "",
        "## fallback 使用情况",
        "",
        "\n".join(fallback_lines) if fallback_lines else "未使用 fallback。",
        "",
        "## 样本缺失计数",
        "",
        markdown_table(pd.DataFrame([{"item": key, "count": value} for key, value in counts.items()])),
        "",
        "## phi0 估计状态",
        "",
    ]
    if calibration["calibration_valid"]:
        lines.append("phi0 已成功估计。")
    else:
        lines.append(f"无法估计 phi0：{calibration['reason_if_invalid']}")
        if not present["tax_rate_or_tax_and_pretax"]:
            lines.append("- 缺少 tax_rate，或缺少 income tax expense 与 pretax income 的组合，无法计算 tau_it。")
        if not present["debt_measure"]:
            lines.append("- 缺少有息债务、总债务、总负债，且无法由 total assets - net assets 推导 D_it。")
        if not present["debt_cost_or_interest_and_lagged_debt"]:
            lines.append("- 缺少 annualized debt_cost，或缺少 interest/finance expense，无法计算 r_it。")
        if calibration.get("n_obs", 0) < MIN_CALIBRATION_OBS:
            lines.append("- 可用于 y_it = Delta d_obs / Delta years 的动态调整回归样本不足。")
    lines.extend(
        [
            "",
            "## 所选数据源行数",
            "",
            markdown_table(
                pd.DataFrame(
                    [{"source": key, "raw_rows": value, "duplicate_rows": build_info.get("source_duplicate_rows", {}).get(key, 0)} for key, value in build_info.get("source_row_counts", {}).items()]
                )
            ),
        ]
    )
    (output_dir / "missing_data_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_quality_report(result: pd.DataFrame, calibration: dict[str, Any], output_dir: Path) -> None:
    total_rows = len(result)
    valid_target = result["optimal_debt_ratio_raw"].notna() if "optimal_debt_ratio_raw" in result else pd.Series(dtype=bool)
    clipped_ratio = (
        float(result.loc[valid_target, "is_optimal_debt_ratio_clipped"].mean())
        if total_rows and int(valid_target.sum()) > 0
        else np.nan
    )
    zero_target_ratio = percent_nonzero_zero(result["optimal_debt_ratio"]) if "optimal_debt_ratio" in result else np.nan
    summary = {
        "企业数量": int(result["firm_id"].nunique()) if total_rows else 0,
        "期间数量": int(result["period_date"].nunique()) if total_rows else 0,
        "样本总行数": int(total_rows),
        "可计算 d_obs 的样本数": int(result["observed_debt_ratio"].notna().sum()) if total_rows else 0,
        "可计算 tau 的样本数": int(result["tax_rate"].notna().sum()) if total_rows else 0,
        "可计算 r 的样本数": int(result["debt_cost"].notna().sum()) if total_rows else 0,
        "可计算 C_core 的样本数": int(result["C_core"].notna().sum()) if total_rows else 0,
        "可用于动态调整校准的样本数": int(calibration.get("n_obs", 0)),
        "d* 被裁剪到上界的比例": clipped_ratio,
        "d*=0 的比例": zero_target_ratio,
        "tax_rate=0 的比例": percent_nonzero_zero(result["tax_rate"]) if total_rows else np.nan,
        "debt_cost=0 的比例": percent_nonzero_zero(result["debt_cost"]) if total_rows else np.nan,
    }
    distribution = distribution_table(
        result,
        ["tax_rate", "debt_cost", "observed_debt_ratio", "optimal_debt_ratio", "leverage_gap"],
    )
    warnings = []
    if clipped_ratio == clipped_ratio and clipped_ratio > 0.20:
        warnings.append(f"- 大量样本的 d* 被裁剪到上界：比例 {clipped_ratio:.2%}，超过 20%。")
    tax_zero = summary["tax_rate=0 的比例"]
    if tax_zero == tax_zero and tax_zero > 0.20:
        warnings.append(f"- 大量样本 tax_rate 为 0：比例 {tax_zero:.2%}。")
    debt_cost_zero = summary["debt_cost=0 的比例"]
    if debt_cost_zero == debt_cost_zero and debt_cost_zero > 0.20:
        warnings.append(f"- 大量样本 debt_cost 为 0：比例 {debt_cost_zero:.2%}。")
    flag_counts = pd.DataFrame(columns=["flag", "count"])
    if total_rows and "data_quality_flags" in result.columns:
        exploded_flags = result["data_quality_flags"].fillna("").str.split(";").explode()
        exploded_flags = exploded_flags[(exploded_flags != "") & (exploded_flags != "ok")]
        if not exploded_flags.empty:
            flag_counts = exploded_flags.value_counts().rename_axis("flag").reset_index(name="count")
            if "annualized_debt_cost_above_100pct" in set(flag_counts["flag"]):
                n_extreme = int(flag_counts.loc[flag_counts["flag"].eq("annualized_debt_cost_above_100pct"), "count"].iloc[0])
                warnings.append(f"- 有 {n_extreme} 行 annualized debt_cost > 100%，这些行保留但已在 data_quality_flags 标记。")
            if "missing_period_length_for_debt_cost" in set(flag_counts["flag"]):
                n_unknown_period = int(flag_counts.loc[flag_counts["flag"].eq("missing_period_length_for_debt_cost"), "count"].iloc[0])
                warnings.append(f"- 有 {n_unknown_period} 行无法识别期间长度，主要来自非季末/半年末/年末日期，无法计算债务成本。")

    lines = [
        "# 数据质量报告",
        "",
        "## 总览",
        "",
        markdown_table(pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()])),
        "",
        "## 分布统计",
        "",
        markdown_table(distribution),
        "",
        "## 异常提示",
        "",
        "\n".join(warnings) if warnings else "未触发比例型异常提示；逐行异常请查看 data_quality_flags。",
        "",
        "## data_quality_flags 计数",
        "",
        markdown_table(flag_counts.head(30)),
    ]
    (output_dir / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class RobustVariantConfig:
    name: str
    label: str
    filename: str
    standard_periods_only: bool
    annual_only: bool
    allow_finance_expense_fallback: bool
    force_finance_expense_proxy: bool = False
    r_upper: float = 0.30
    tau_upper: float = 0.50
    output_clean_sample_only: bool = True


ROBUST_VARIANTS = [
    RobustVariantConfig(
        name="A_clean_main",
        label="主结果 A：标准报告期、直接利息费用、干净校准样本",
        filename="variant_A_clean_main_results.csv",
        standard_periods_only=True,
        annual_only=False,
        allow_finance_expense_fallback=False,
    ),
    RobustVariantConfig(
        name="B_finance_fallback",
        label="稳健性 B：标准报告期、允许财务费用 fallback、干净校准样本",
        filename="variant_B_finance_fallback_results.csv",
        standard_periods_only=True,
        annual_only=False,
        allow_finance_expense_fallback=True,
    ),
    RobustVariantConfig(
        name="C_annual_direct",
        label="稳健性 C：仅 12-31 年报、直接利息费用、干净校准样本",
        filename="variant_C_annual_results.csv",
        standard_periods_only=True,
        annual_only=True,
        allow_finance_expense_fallback=False,
    ),
    RobustVariantConfig(
        name="D_finance_proxy_consistent",
        label="Robustness D: standard periods, consistently use finance expense B001211000 as the interest-expense proxy",
        filename="variant_D_finance_proxy_consistent_results.csv",
        standard_periods_only=True,
        annual_only=False,
        allow_finance_expense_fallback=True,
        force_finance_expense_proxy=True,
    ),
]

ROBUST_VARIANTS = [
    RobustVariantConfig(
        name="A_clean_main",
        label="主结果 A：标准报告期、直接利息费用、干净校准样本",
        filename="variant_A_clean_main_results.csv",
        standard_periods_only=True,
        annual_only=False,
        allow_finance_expense_fallback=False,
    ),
    RobustVariantConfig(
        name="B_finance_fallback",
        label="稳健性 B：标准报告期、允许财务费用 fallback、干净校准样本",
        filename="variant_B_finance_fallback_results.csv",
        standard_periods_only=True,
        annual_only=False,
        allow_finance_expense_fallback=True,
    ),
    RobustVariantConfig(
        name="C_annual_direct",
        label="稳健性 C：仅 12-31 年报、直接利息费用、干净校准样本",
        filename="variant_C_annual_results.csv",
        standard_periods_only=True,
        annual_only=True,
        allow_finance_expense_fallback=False,
    ),
    RobustVariantConfig(
        name="D_finance_proxy_consistent",
        label="Robustness D: standard periods, consistently use finance expense B001211000 as the interest-expense proxy",
        filename="variant_D_finance_proxy_consistent_results.csv",
        standard_periods_only=True,
        annual_only=False,
        allow_finance_expense_fallback=True,
        force_finance_expense_proxy=True,
    ),
]


def is_standard_report_date(period_date: pd.Series) -> pd.Series:
    dates = pd.to_datetime(period_date, errors="coerce")
    return dates.dt.strftime("%m-%d").isin(["03-31", "06-30", "09-30", "12-31"])


def fiscal_coverage_years(period_date: pd.Series) -> pd.Series:
    dates = pd.to_datetime(period_date, errors="coerce")
    month_day = dates.dt.strftime("%m-%d")
    mapping = {"03-31": 0.25, "06-30": 0.50, "09-30": 0.75, "12-31": 1.00}
    return month_day.map(mapping).astype(float)


def construct_debt_measure(out: pd.DataFrame, flags: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    total_debt_used = pd.Series(np.nan, index=out.index, dtype=float)
    debt_source_flag = pd.Series("", index=out.index, dtype=object)
    if "total_debt" in out.columns and out["total_debt"].notna().any():
        total_debt_used = out["total_debt"].astype(float)
        debt_source_flag.loc[total_debt_used.notna()] = "direct_total_debt_or_interest_bearing_debt"

    component_cols = [col for col in DEBT_COMPONENT_FIELDS if col in out.columns and out[col].notna().any()]
    if component_cols:
        component_debt = out[component_cols].sum(axis=1, min_count=1)
        fill_mask = total_debt_used.isna() & component_debt.notna()
        total_debt_used.loc[fill_mask] = component_debt.loc[fill_mask]
        debt_source_flag.loc[fill_mask] = "sum_interest_bearing_debt_components"

    if "total_liabilities" in out.columns and out["total_liabilities"].notna().any():
        fill_mask = total_debt_used.isna() & out["total_liabilities"].notna()
        total_debt_used.loc[fill_mask] = out.loc[fill_mask, "total_liabilities"]
        debt_source_flag.loc[fill_mask] = "total_liabilities_proxy"
        flags = append_flag(flags, fill_mask, "observed_leverage_used_total_liabilities_proxy")

    if "total_assets" in out.columns and "net_assets" in out.columns:
        derived = out["total_assets"] - out["net_assets"]
        negative = derived < -1e-8 * out["total_assets"].abs().fillna(0)
        fill_mask = total_debt_used.isna() & derived.notna() & ~negative
        derived = derived.mask(derived.abs() < 1e-8, 0.0)
        total_debt_used.loc[fill_mask] = derived.loc[fill_mask]
        debt_source_flag.loc[fill_mask] = "total_liabilities_proxy_from_total_assets_minus_net_assets"
        flags = append_flag(flags, fill_mask, "observed_leverage_used_total_assets_minus_net_assets_proxy")
        flags = append_flag(flags, total_debt_used.isna() & negative, "negative_derived_liabilities")

    return total_debt_used, debt_source_flag.replace("", np.nan), flags


def compute_tax_rate_series(out: pd.DataFrame, flags: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    if "tax_rate" in out.columns and out["tax_rate"].notna().any():
        raw = pd.to_numeric(out["tax_rate"], errors="coerce")
        finite = raw[np.isfinite(raw)]
        if not finite.empty and finite.abs().median() > 1 and finite.abs().median() <= 100:
            raw = raw / 100.0
            flags = append_flag(flags, out["tax_rate"].notna(), "tax_rate_interpreted_as_percent")
    elif "tax_expense" in out.columns and "pretax_income" in out.columns:
        raw = pd.to_numeric(out["tax_expense"], errors="coerce") / pd.to_numeric(out["pretax_income"], errors="coerce")
        non_positive_pretax = out["pretax_income"].notna() & (out["pretax_income"] <= 0)
        raw = raw.mask(non_positive_pretax, 0.0)
        flags = append_flag(flags, non_positive_pretax, "pretax_income_non_positive_tax_rate_set_to_0")
    else:
        raw = pd.Series(np.nan, index=out.index)
        flags = append_flag(flags, pd.Series(True, index=out.index), "missing_tax_rate_inputs")
    clipped = raw.clip(lower=0.0, upper=1.0)
    flags = append_flag(flags, raw.notna() & (raw != clipped), "tax_rate_clipped_to_0_1")
    flags = append_flag(flags, clipped.isna(), "cannot_compute_tax_rate")
    return raw, clipped, flags


def compute_period_interest_and_cost(out: pd.DataFrame, config: RobustVariantConfig, flags: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    work = out.sort_values(["firm_id", "period_date", "period_key"]).copy()
    if "interest_expense" in work.columns:
        ytd_interest = pd.to_numeric(work["interest_expense"], errors="coerce")
    else:
        ytd_interest = pd.Series(np.nan, index=work.index)
    work["interest_source_flag"] = np.where(ytd_interest.notna(), "direct_interest_expense", "missing")
    flags = append_flag(flags, ytd_interest.isna(), "missing_direct_interest_expense")

    if config.force_finance_expense_proxy:
        if "finance_expense" in work.columns:
            finance_interest = pd.to_numeric(work["finance_expense"], errors="coerce")
        else:
            finance_interest = pd.Series(np.nan, index=work.index)
        direct_available = ytd_interest.notna()
        ytd_interest = finance_interest
        work["interest_source_flag"] = np.where(finance_interest.notna(), "finance_expense_proxy", "missing")
        flags = append_flag(flags, finance_interest.notna(), "finance_expense_used_as_interest_expense_proxy")
        flags = append_flag(flags, finance_interest.isna(), "missing_finance_expense_for_forced_proxy")
        flags = append_flag(
            flags,
            direct_available & finance_interest.notna(),
            "direct_interest_expense_ignored_for_consistent_finance_proxy",
        )
    elif config.allow_finance_expense_fallback and "finance_expense" in work.columns:
        fallback_mask = ytd_interest.isna() & work["finance_expense"].notna()
        ytd_interest.loc[fallback_mask] = pd.to_numeric(work.loc[fallback_mask, "finance_expense"], errors="coerce")
        work.loc[fallback_mask, "interest_source_flag"] = "finance_expense_proxy"
        flags = append_flag(flags, fallback_mask, "finance_expense_used_as_interest_expense_proxy")

    work["interest_expense_ytd_used"] = ytd_interest
    work["fiscal_year"] = work["period_date"].dt.year
    work["fiscal_coverage_years"] = fiscal_coverage_years(work["period_date"])

    if config.annual_only:
        period_interest = ytd_interest.where(work["period_date"].dt.strftime("%m-%d").eq("12-31"))
        period_years = pd.Series(1.0, index=work.index).where(period_interest.notna())
    else:
        prev_ytd = ytd_interest.groupby([work["firm_id"], work["fiscal_year"]]).shift(1)
        prev_date = work.groupby(["firm_id", "fiscal_year"])["period_date"].shift(1)
        has_prev = prev_ytd.notna() & prev_date.notna()
        period_interest = ytd_interest.where(~has_prev, ytd_interest - prev_ytd)
        period_years = work["fiscal_coverage_years"].where(~has_prev, (work["period_date"] - prev_date).dt.days / 365.25)

    work["period_interest_expense"] = period_interest
    work["period_interest_years"] = period_years
    flags = append_flag(flags, period_interest.isna(), "missing_period_interest_expense")
    flags = append_flag(flags, period_interest.notna() & (period_interest < 0), "negative_period_interest_expense")
    flags = append_flag(flags, period_years.isna() | (period_years <= 0), "missing_period_length_for_debt_cost")

    lag_debt = work.groupby("firm_id")["total_debt_used"].shift(1)
    avg_debt = (lag_debt + work["total_debt_used"]) / 2.0
    work["average_debt_for_cost"] = avg_debt
    raw_period_cost = period_interest / avg_debt
    valid = (avg_debt > 0) & (period_years > 0) & raw_period_cost.notna()
    work["debt_cost_raw"] = raw_period_cost.where(valid)
    work["debt_cost"] = (raw_period_cost / period_years).where(valid)
    flags = append_flag(flags, lag_debt.isna() | (lag_debt <= 0) | (avg_debt <= 0), "missing_or_non_positive_average_debt_for_debt_cost")
    flags = append_flag(flags, work["debt_cost"].isna(), "cannot_compute_debt_cost")
    flags = append_flag(flags, work["debt_cost"].notna() & (work["debt_cost"] <= 0), "non_positive_debt_cost")
    flags = append_flag(flags, work["debt_cost"].notna() & (work["debt_cost"] >= config.r_upper), f"debt_cost_outside_0_{config.r_upper:g}")
    flags = append_flag(flags, work["debt_cost"].notna() & (work["debt_cost"] > 1), "annualized_debt_cost_above_100pct")
    return work, flags.loc[work.index]


def calibrate_phi0_with_mask(out: pd.DataFrame, eta: float, target_horizon_years: float, min_calibration_obs: int, current_mask: pd.Series) -> dict[str, Any]:
    work = out[["firm_id", "period_date", "observed_debt_ratio", "C_core"]].copy()
    work["current_clean_mask"] = current_mask.astype(bool)
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
        & (work["delta_years"] > 0)
        & (work["delta_years"] <= 1.1)
        & np.isfinite(work["y"])
        & np.isfinite(work["C_core"])
        & np.isfinite(work["observed_debt_ratio"])
    )
    reg = work.loc[valid, ["y", "C_core", "observed_debt_ratio"]].copy()
    n_obs = int(len(reg))
    b_c = np.nan
    b_d = np.nan
    kappa_hat = np.nan
    s_hat = np.nan
    phi0_hat = np.nan
    calibration_valid = False
    reason = ""
    if n_obs < min_calibration_obs:
        reason = f"可用于稳健动态校准的样本数 {n_obs} 少于 min_calibration_obs={min_calibration_obs}。"
    else:
        try:
            coef, *_ = np.linalg.lstsq(reg[["C_core", "observed_debt_ratio"]].to_numpy(dtype=float), reg["y"].to_numpy(dtype=float), rcond=None)
            b_c = float(coef[0])
            b_d = float(coef[1])
            if not np.isfinite(b_c) or not np.isfinite(b_d):
                reason = "无截距回归系数不是有限数。"
            elif b_c <= 0:
                reason = f"动态调整回归得到 b_C={b_c:.6g} <= 0。"
            elif b_d >= 0:
                reason = f"动态调整回归得到 b_d={b_d:.6g} >= 0。"
            else:
                kappa_hat = -b_d
                s_hat = b_c / (-b_d)
                phi0_hat = ((-b_d) / b_c) ** (eta - 1.0)
                if not np.isfinite(phi0_hat) or phi0_hat <= 0:
                    reason = f"估计出的 phi0={phi0_hat} <= 0 或非有限。"
                else:
                    calibration_valid = True
                    reason = "ok"
        except Exception as exc:
            reason = f"无截距回归失败：{type(exc).__name__}: {exc}"
    return {
        "eta": eta,
        "T": target_horizon_years,
        "n_obs": n_obs,
        "b_C": b_c,
        "b_d": b_d,
        "kappa_hat": kappa_hat,
        "s_hat": s_hat,
        "phi0_hat": phi0_hat,
        "calibration_valid": bool(calibration_valid),
        "reason_if_invalid": "" if calibration_valid else reason,
        "calibration_sample_rows": int(valid.sum()),
    }


def normalize_robust_calibration_reason(calibration: dict[str, Any], min_calibration_obs: int) -> dict[str, Any]:
    if calibration.get("calibration_valid"):
        calibration["reason_if_invalid"] = ""
        return calibration

    n_obs = int(calibration.get("n_obs") or 0)
    b_c = calibration.get("b_C")
    b_d = calibration.get("b_d")
    phi0_hat = calibration.get("phi0_hat")
    if n_obs < min_calibration_obs:
        reason = f"可用于稳健动态校准的样本数 {n_obs} 少于 min_calibration_obs={min_calibration_obs}。"
    elif not np.isfinite(b_c) or not np.isfinite(b_d):
        reason = "无截距回归系数不是有限数。"
    elif b_c <= 0:
        reason = f"动态调整回归得到 b_C={b_c:.6g} <= 0。"
    elif b_d >= 0:
        reason = f"动态调整回归得到 b_d={b_d:.6g} >= 0。"
    elif not np.isfinite(phi0_hat) or phi0_hat <= 0:
        reason = f"估计出的 phi0={phi0_hat} <= 0 或非有限。"
    else:
        reason = str(calibration.get("reason_if_invalid") or "校准未通过，但未识别到具体原因。")
    calibration["reason_if_invalid"] = reason
    return calibration


def compute_robust_variant(panel: pd.DataFrame, config: RobustVariantConfig, eta: float, target_horizon_years: float, min_calibration_obs: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = panel.copy()
    for col in ["firm_id", "period_date", "period_key", *VALUE_FIELDS]:
        if col not in out.columns:
            out[col] = np.nan
    out["period_date"] = pd.to_datetime(out["period_date"], errors="coerce")
    flags = pd.Series("", index=out.index, dtype=object)
    out["is_standard_report_date"] = is_standard_report_date(out["period_date"])
    if config.standard_periods_only:
        out = out[out["is_standard_report_date"]].copy()
        flags = flags.loc[out.index].copy()
    if config.annual_only:
        out = out[out["period_date"].dt.strftime("%m-%d").eq("12-31")].copy()
        flags = flags.loc[out.index].copy()
    out = out.sort_values(["firm_id", "period_date", "period_key"]).reset_index(drop=True)
    flags = pd.Series("", index=out.index, dtype=object)

    out["period_type"] = complete_period_types(out)
    if not config.annual_only:
        is_half_year_date = out["period_date"].dt.strftime("%m-%d").eq("06-30")
        out["period_type"] = np.where(is_half_year_date, "quarterly_or_semiannual_ytd_increment", out["period_type"])
    out["total_debt_used"], out["debt_source_flag"], flags = construct_debt_measure(out, flags)
    valid_observed = (out["total_assets"] > 0) & (out["total_debt_used"] >= 0)
    out["observed_debt_ratio"] = np.where(valid_observed, out["total_debt_used"] / out["total_assets"], np.nan)
    flags = append_flag(flags, out["total_assets"].isna(), "missing_total_assets")
    flags = append_flag(flags, out["total_assets"].notna() & (out["total_assets"] <= 0), "non_positive_total_assets")
    flags = append_flag(flags, out["total_debt_used"].isna(), "missing_debt_measure")
    flags = append_flag(flags, out["observed_debt_ratio"].notna() & (out["observed_debt_ratio"] > 1), "observed_debt_ratio_above_1")

    out["tax_rate_raw"], out["tax_rate"], flags = compute_tax_rate_series(out, flags)
    out, flags = compute_period_interest_and_cost(out, config, flags)
    out = out.reset_index(drop=True)
    flags = flags.reset_index(drop=True)

    a = 1.0 / (eta - 1.0)
    out["target_horizon_years"] = target_horizon_years
    out["eta"] = eta
    numerator_core = out["tax_rate"] * ((1.0 + out["debt_cost"]) ** target_horizon_years - 1.0)
    valid_core = out["tax_rate"].notna() & out["debt_cost"].notna() & (out["debt_cost"] > -1) & (numerator_core >= 0) & np.isfinite(numerator_core)
    out["C_core"] = np.where(valid_core, (numerator_core / (target_horizon_years * eta)) ** a, np.nan)
    flags = append_flag(flags, out["C_core"].isna(), "cannot_compute_C_core")

    clean_mask = (
        out["is_standard_report_date"]
        & out["observed_debt_ratio"].between(0, 1, inclusive="neither")
        & out["tax_rate"].between(0, config.tau_upper, inclusive="neither")
        & out["debt_cost"].between(0, config.r_upper, inclusive="neither")
        & out["C_core"].notna()
    )
    if config.annual_only:
        clean_mask = clean_mask & out["period_date"].dt.strftime("%m-%d").eq("12-31")
    if not config.allow_finance_expense_fallback:
        clean_mask = clean_mask & out["interest_source_flag"].eq("direct_interest_expense")
    out["calibration_sample_flag"] = clean_mask
    flags = append_flag(flags, ~clean_mask, "excluded_from_robust_calibration")

    calibration = calibrate_phi0_with_mask(out, eta, target_horizon_years, min_calibration_obs, clean_mask)
    calibration = normalize_robust_calibration_reason(calibration, min_calibration_obs)
    out["phi0_hat"] = calibration.get("phi0_hat", np.nan)
    out["optimal_debt_ratio_raw"] = np.nan
    out["optimal_debt_ratio"] = np.nan
    out["is_optimal_debt_ratio_clipped"] = False
    out["leverage_gap_raw"] = np.nan
    out["leverage_gap"] = np.nan
    if calibration["calibration_valid"]:
        numerator = out["tax_rate"] * ((1.0 + out["debt_cost"]) ** target_horizon_years - 1.0)
        denominator = calibration["phi0_hat"] * target_horizon_years * eta
        valid_target = out["tax_rate"].notna() & out["debt_cost"].notna() & (out["debt_cost"] > -1) & (numerator >= 0) & (denominator > 0)
        raw_target = (numerator / denominator) ** a
        out["optimal_debt_ratio_raw"] = raw_target.where(valid_target)
        out["optimal_debt_ratio"] = out["optimal_debt_ratio_raw"].clip(lower=0.0, upper=DEBT_RATIO_UPPER)
        out["is_optimal_debt_ratio_clipped"] = out["optimal_debt_ratio_raw"].notna() & (
            (out["optimal_debt_ratio_raw"] < 0) | (out["optimal_debt_ratio_raw"] > DEBT_RATIO_UPPER)
        )
        out["leverage_gap_raw"] = out["observed_debt_ratio"] - out["optimal_debt_ratio_raw"]
        out["leverage_gap"] = out["observed_debt_ratio"] - out["optimal_debt_ratio"]
        flags = append_flag(flags, out["optimal_debt_ratio_raw"].isna(), "cannot_compute_optimal_debt_ratio")
    else:
        flags = append_flag(flags, pd.Series(True, index=out.index), "phi0_calibration_invalid_no_optimal_debt_ratio")

    out["leverage_status"] = np.select(
        [
            out["leverage_gap"].isna(),
            out["leverage_gap"].abs() <= NEAR_OPTIMAL_TOL,
            out["leverage_gap"] > NEAR_OPTIMAL_TOL,
            out["leverage_gap"] < -NEAR_OPTIMAL_TOL,
        ],
        ["missing", "near_optimal", "over_levered", "under_levered"],
        default="missing",
    )
    out["model_variant"] = config.name
    out["variant_label"] = config.label
    out["data_quality_flags"] = flags.replace("", "ok")

    if config.output_clean_sample_only:
        out = out[out["calibration_sample_flag"]].copy()

    result_cols = [
        "model_variant",
        "firm_id",
        "period_date",
        "period_type",
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
    for col in result_cols:
        if col not in out.columns:
            out[col] = np.nan
    calibration.update(
        {
            "variant": config.name,
            "label": config.label,
            "allow_finance_expense_fallback": config.allow_finance_expense_fallback,
            "force_finance_expense_proxy": config.force_finance_expense_proxy,
            "annual_only": config.annual_only,
            "r_filter": f"0 < r < {config.r_upper:g}",
            "tau_filter": f"0 < tau < {config.tau_upper:g}",
            "d_obs_filter": "0 < d_obs < 1",
            "output_rows": int(len(out)),
            "dstar_count": int(out["optimal_debt_ratio"].notna().sum()),
            "gap_count": int(out["leverage_gap"].notna().sum()),
            "dstar_median": float(out["optimal_debt_ratio"].median()) if out["optimal_debt_ratio"].notna().any() else np.nan,
            "gap_median": float(out["leverage_gap"].median()) if out["leverage_gap"].notna().any() else np.nan,
            "status_counts": json.dumps(out["leverage_status"].value_counts(dropna=False).to_dict(), ensure_ascii=False),
        }
    )
    return out[result_cols], calibration


def robust_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    return distribution_table(df, ["observed_debt_ratio", "tax_rate", "debt_cost", "C_core", "optimal_debt_ratio", "leverage_gap"])


def write_robustness_report(variant_outputs: dict[str, pd.DataFrame], calibrations: list[dict[str, Any]], output_dir: Path) -> None:
    cal_df = pd.DataFrame(calibrations)
    lines = [
        "# 稳健校准结果报告",
        "",
        "本报告追加三套结果，用于处理未清洗基线中 `phi0` 过大、`d*` 坍缩到 0 附近的问题。理论公式未改动，调整只发生在债务成本口径、期间流量处理和校准样本过滤层。",
        "",
        "## 三套设定",
        "",
        "| 版本 | 设定 |",
        "| --- | --- |",
        "| A | 标准报告期；只用直接利息费用 `B001211101`；期间增量利息；简单年化；校准样本满足 `0<r<0.30, 0<tau<0.5, 0<d_obs<1` |",
        "| B | 在 A 的基础上，允许用财务费用 `B001211000` 补直接利息费用缺失值 |",
        "| C | 只用 12-31 年报；只用直接利息费用；年度费用不做季度增量拆分；同样使用干净校准过滤 |",
        "| D | 标准报告期；整段时间一致使用财务费用 `B001211000` 作为利息费用 proxy，避免 2018 前后因字段来源切换产生定义跳变 |",
        "",
        "## 校准汇总",
        "",
        markdown_table(
            cal_df[
                [
                    "variant",
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
            ]
        ),
        "",
    ]
    for cal in calibrations:
        name = cal["variant"]
        df = variant_outputs[name]
        lines.extend(
            [
                f"## {name}",
                "",
                cal["label"],
                "",
                "杠杆状态分布：",
                "",
                markdown_table(df["leverage_status"].value_counts(dropna=False).rename_axis("leverage_status").reset_index(name="count")),
                "",
                "分布统计：",
                "",
                markdown_table(robust_distribution_table(df)),
                "",
            ]
        )
    lines.extend(
        [
            "## 使用建议",
            "",
            "若 A 版的 `phi0` 和 `d*` 分布明显比未清洗基线稳定，应优先把 A 作为主结果。B 用于检验财务费用 fallback 是否改变结论；C 用于判断季度累计口径和年化处理是否是主要问题来源。",
            "",
            "如果三版仍然出现 `b_C` 接近 0 或 `phi0` 极端放大，下一步应考虑分金融/非金融样本、分年度校准，或对动态调整方程加入更严格的 winsorize 处理，而不是修改 d* 的理论公式。",
        ]
    )
    (output_dir / "robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_robustness_report(variant_outputs: dict[str, pd.DataFrame], calibrations: list[dict[str, Any]], output_dir: Path) -> None:
    cal_df = pd.DataFrame(calibrations)
    summary_cols = [
        "variant",
        "allow_finance_expense_fallback",
        "force_finance_expense_proxy",
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
    summary_cols = [col for col in summary_cols if col in cal_df.columns]
    lines = [
        "# 稳健校准结果报告",
        "",
        "本报告追加三套结果，用于处理未清洗基线中 `phi0` 过大、`d*` 坍缩到 0 附近的问题。理论公式未改动，调整只发生在债务成本口径、期间流量处理和校准样本过滤层。",
        "",
        "## 三套设定",
        "",
        "| 版本 | 设定 |",
        "| --- | --- |",
        "| A | 标准报告期；只用直接利息费用 `B001211101`；利润表 YTD 转期间增量；简单年化；校准样本满足 `0<r<0.30, 0<tau<0.5, 0<d_obs<1` |",
        "| B | 在 A 的基础上，允许用财务费用 `B001211000` 补直接利息费用缺失值；其他过滤相同 |",
        "| C | 只用 12-31 年报；只用直接利息费用；年度费用不做季度增量拆分；校准过滤同 A |",
        "| D | 标准报告期；整段时间一致使用财务费用 `B001211000` 作为利息费用 proxy，避免 2018 前后因字段来源切换产生定义跳变 |",
        "",
        "## 校准汇总",
        "",
        markdown_table(cal_df[summary_cols]) if summary_cols else "(无校准结果)",
        "",
    ]
    for cal in calibrations:
        name = cal["variant"]
        df = variant_outputs[name]
        lines.extend(
            [
                f"## {name}",
                "",
                str(cal.get("label", "")),
                "",
                "杠杆状态分布：",
                "",
                markdown_table(df["leverage_status"].value_counts(dropna=False).rename_axis("leverage_status").reset_index(name="count")),
                "",
                "分布统计：",
                "",
                markdown_table(robust_distribution_table(df)),
                "",
            ]
        )
    lines.extend(
        [
            "## 使用建议",
            "",
            "优先把 A 作为主结果候选；B 用于检验财务费用 fallback 是否显著改变结论；C 用于判断季度累计口径和年化处理是否是主要问题来源。",
            "",
            "如果三版仍出现 `b_C` 接近 0 或 `phi0` 极端放大，下一步应考虑分金融/非金融样本、分年度校准，或在动态调整方程前进一步处理异常杠杆变化，而不是修改 `d*` 的理论公式。",
        ]
    )
    (output_dir / "robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_robust_variants(panel: pd.DataFrame, eta: float, target_horizon_years: float, min_calibration_obs: int, output_dir: Path) -> None:
    outputs: dict[str, pd.DataFrame] = {}
    calibrations: list[dict[str, Any]] = []
    for config in ROBUST_VARIANTS:
        result, calibration = compute_robust_variant(panel, config, eta, target_horizon_years, min_calibration_obs)
        result.to_csv(output_dir / config.filename, index=False, encoding="utf-8-sig")
        outputs[config.name] = result
        calibrations.append(calibration)
    cal_df = pd.DataFrame(calibrations)
    leading_cols = ["variant", "label", "eta", "T", "n_obs", "b_C", "b_d", "kappa_hat", "s_hat", "phi0_hat", "calibration_valid", "reason_if_invalid"]
    cal_df = cal_df[[col for col in leading_cols if col in cal_df.columns] + [col for col in cal_df.columns if col not in leading_cols]]
    cal_df.to_csv(output_dir / "phi0_calibration_robust_variants.csv", index=False, encoding="utf-8-sig")
    write_robustness_report(outputs, calibrations, output_dir)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_empty_outputs(reason: str, profiles: list[SourceProfile], output_dir: Path) -> None:
    columns = [
        "firm_id",
        "period_date",
        "period_type",
        "total_assets",
        "total_debt_used",
        "debt_source_flag",
        "observed_debt_ratio",
        "tax_rate_raw",
        "tax_rate",
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
        "data_quality_flags",
    ]
    result = pd.DataFrame(columns=columns)
    result.to_csv(output_dir / "optimal_leverage_results.csv", index=False, encoding="utf-8-sig")
    calibration = {
        "eta": np.nan,
        "T": np.nan,
        "n_obs": 0,
        "b_C": np.nan,
        "b_d": np.nan,
        "kappa_hat": np.nan,
        "s_hat": np.nan,
        "phi0_hat": np.nan,
        "calibration_valid": False,
        "reason_if_invalid": reason,
    }
    write_calibration_outputs(calibration, output_dir)
    write_quality_report(result, calibration, output_dir)
    write_missing_report(result, {}, [], {"source_duplicate_rows": {}, "source_row_counts": {}}, calibration, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute structural trade-off optimal leverage d* and leverage Gap.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--target-horizon-years", type=float, default=1.0)
    parser.add_argument("--min-calibration-obs", type=int, default=MIN_CALIBRATION_OBS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    print(f"扫描数据目录: {args.source_root}")
    profiles = scan_sources(args.source_root)
    write_manifest(profiles, args.output_dir / "source_data_manifest.csv")
    selected_profiles, mapping = select_profiles(profiles)
    write_field_mapping(mapping, selected_profiles, args.output_dir / "field_mapping.json")

    if not selected_profiles:
        reason = "未找到同时包含企业 ID 和时间字段的可用财务面板数据。"
        write_empty_outputs(reason, profiles, args.output_dir)
        print(reason)
        print(f"主结果文件路径: {args.output_dir / 'optimal_leverage_results.csv'}")
        print(f"缺失数据报告路径: {args.output_dir / 'missing_data_report.md'}")
        return 0

    print("使用的数据文件:")
    for profile in selected_profiles:
        print(f"- {profile.source.ref}")
    print("识别出的字段映射:")
    for field, item in mapping.items():
        print(f"- {field}: {item['column']} @ {item['source']}")

    panel, build_info = build_panel(selected_profiles, mapping)
    if panel.empty:
        reason = "所选数据源读取后没有可用 firm_id-period_date 面板行。"
        write_empty_outputs(reason, profiles, args.output_dir)
        print(reason)
        return 0

    result, calibration = compute_model(
        panel,
        eta=args.eta,
        target_horizon_years=args.target_horizon_years,
        min_calibration_obs=args.min_calibration_obs,
    )
    result_path = args.output_dir / "optimal_leverage_results.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    write_calibration_outputs(calibration, args.output_dir)
    write_missing_report(result, mapping, selected_profiles, build_info, calibration, args.output_dir)
    write_quality_report(result, calibration, args.output_dir)
    run_robust_variants(
        panel,
        eta=args.eta,
        target_horizon_years=args.target_horizon_years,
        min_calibration_obs=args.min_calibration_obs,
        output_dir=args.output_dir,
    )

    print(f"样本规模: rows={len(result)}, firms={result['firm_id'].nunique()}, periods={result['period_date'].nunique()}")
    print(f"phi0_hat: {calibration.get('phi0_hat')}")
    print(f"校准是否有效: {calibration.get('calibration_valid')} ({calibration.get('reason_if_invalid') or 'ok'})")
    print(f"主结果文件路径: {result_path}")
    print(f"稳健性校准路径: {args.output_dir / 'phi0_calibration_robust_variants.csv'}")
    print(f"稳健性报告路径: {args.output_dir / 'robustness_report.md'}")
    print(f"缺失数据报告路径: {args.output_dir / 'missing_data_report.md'}")
    print(f"校准结果文件路径: {args.output_dir / 'phi0_calibration.csv'}")
    print(f"数据质量报告路径: {args.output_dir / 'data_quality_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
