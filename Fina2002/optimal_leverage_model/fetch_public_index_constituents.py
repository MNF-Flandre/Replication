from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "public_index_data"

DEFAULT_INDEX_CODES = {
    "000300": "CSI300 / 沪深300",
    "000905": "CSI500 / 中证500",
    "000906": "CSI800 / 中证800",
    "000852": "CSI1000 / 中证1000",
    "932000": "CSI2000 / 中证2000",
    "000510": "CSI A500 / 中证A500",
    "000016": "SSE50 / 上证50",
    "000010": "SSE180 / 上证180",
}

CSI_URLS = {
    "latest_weight": "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/closeweight/{code}closeweight.xls",
    "latest_constituents": "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/{code}cons.xls",
}

WEIGHT_COLUMNS = [
    "date",
    "index_code",
    "index_name",
    "index_name_en",
    "stock_code",
    "stock_name",
    "stock_name_en",
    "exchange",
    "exchange_en",
    "weight_pct",
]

CONSTITUENT_COLUMNS = [
    "date",
    "index_code",
    "index_name",
    "index_name_en",
    "stock_code",
    "stock_name",
    "stock_name_en",
    "exchange",
    "exchange_en",
]


def parse_codes(value: str) -> dict[str, str]:
    if not value.strip():
        return DEFAULT_INDEX_CODES
    result = {}
    for item in value.split(","):
        code = item.strip()
        if code:
            result[code.zfill(6)] = code.zfill(6)
    return result


def fetch_excel(url: str, timeout: int) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.csindex.com.cn/",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    if len(response.content) < 200:
        raise ValueError(f"downloaded file is too small: {len(response.content)} bytes")
    return response.content


def parse_csi_excel(content: bytes, dataset: str) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(content))
    expected = WEIGHT_COLUMNS if dataset == "latest_weight" else CONSTITUENT_COLUMNS
    if len(df.columns) != len(expected):
        raise ValueError(f"unexpected column count {len(df.columns)} for {dataset}")
    df.columns = expected
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").dt.date
    df["index_code"] = df["index_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    df["stock_code"] = df["stock_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    if "weight_pct" in df.columns:
        df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce")
        df["weight"] = df["weight_pct"] / 100.0
    return df


def fetch_one(code: str, dataset: str, timeout: int) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    url = CSI_URLS[dataset].format(code=code)
    row: dict[str, Any] = {
        "index_code_requested": code,
        "dataset": dataset,
        "source": "CSI public static file",
        "url": url,
        "status": "fail",
    }
    try:
        content = fetch_excel(url, timeout)
        df = parse_csi_excel(content, dataset)
        row.update(
            {
                "status": "ok",
                "rows": int(len(df)),
                "date": str(df["date"].dropna().iloc[0]) if df["date"].notna().any() else "",
                "index_code_returned": df["index_code"].dropna().iloc[0] if len(df) else "",
                "index_name": df["index_name"].dropna().iloc[0] if len(df) else "",
                "weight_sum_pct": float(df["weight_pct"].sum()) if "weight_pct" in df.columns else None,
            }
        )
        return df, row
    except Exception as exc:
        row.update({"error": f"{type(exc).__name__}: {str(exc)[:500]}"})
        return None, row


def write_report(output_dir: Path, manifest: pd.DataFrame) -> None:
    ok = manifest[manifest["status"].eq("ok")].copy()
    fail = manifest[~manifest["status"].eq("ok")].copy()
    lines = [
        "# Public Index Constituents Fetch Report",
        "",
        "## Source",
        "",
        "Data are fetched from CSI public static files used by the CSI index pages:",
        "",
        "- latest constituents: `.../autofile/cons/{code}cons.xls`",
        "- latest weights: `.../autofile/closeweight/{code}closeweight.xls`",
        "",
        "These files provide current or most recent constituent details. They are not a full historical daily weight database.",
        "",
        "## Successful Fetches",
        "",
        ok.to_markdown(index=False) if not ok.empty else "(none)",
        "",
        "## Failed Fetches",
        "",
        fail.to_markdown(index=False) if not fail.empty else "(none)",
        "",
        "## Use In This Project",
        "",
        "The downloaded latest-weight files can be used for current constituent inspection or a point-in-time test around the file date. They should not be used to backfill historical index membership in earlier years.",
    ]
    (output_dir / "public_index_fetch_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    codes = parse_codes(args.index_codes)
    rows: list[dict[str, Any]] = []
    normalized_weight_parts: list[pd.DataFrame] = []
    for code, label in codes.items():
        for dataset in ["latest_weight", "latest_constituents"]:
            df, row = fetch_one(code, dataset, args.timeout)
            row["index_label_requested"] = label
            rows.append(row)
            if df is not None:
                df.to_csv(output_dir / f"{code}_{dataset}_csindex_public.csv", index=False, encoding="utf-8-sig")
                if dataset == "latest_weight":
                    normalized = df[["index_code", "date", "stock_code", "stock_name", "weight_pct"]].copy()
                    normalized = normalized.rename(
                        columns={
                            "index_code": "Indexcd",
                            "date": "Enddt",
                            "stock_code": "Stkcd",
                            "stock_name": "Constdnme",
                            "weight_pct": "Weight",
                        }
                    )
                    normalized["source"] = "csindex_public_latest_weight"
                    normalized_weight_parts.append(normalized)
            print(f"{code} {dataset}: {row['status']} rows={row.get('rows', 0)}")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "public_index_fetch_manifest.csv", index=False, encoding="utf-8-sig")
    if normalized_weight_parts:
        combined = pd.concat(normalized_weight_parts, ignore_index=True)
        combined.to_csv(
            output_dir / "public_latest_index_weights_IDX_Smprat_format.csv",
            index=False,
            encoding="utf-8-sig",
        )
    write_report(output_dir, manifest)
    print(f"Manifest: {output_dir / 'public_index_fetch_manifest.csv'}")
    print(f"Report: {output_dir / 'public_index_fetch_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest public CSI index constituents and weights.")
    parser.add_argument("--index-codes", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
