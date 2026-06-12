from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from qlib_framework.settings import OUTPUT_DIR, PROVIDER_URI


DEFAULT_FIELDS = ["$open", "$close", "$volume", "$amount", "$return", "$factor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that project-local Qlib data can be initialized and read.")
    parser.add_argument("--provider-uri", type=Path, default=PROVIDER_URI)
    parser.add_argument("--start-time", default="2020-01-02")
    parser.add_argument("--end-time", default="2020-01-10")
    parser.add_argument("--instruments", nargs="*", default=["SH600000", "SZ000001", "SH600519"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider_uri = args.provider_uri.expanduser().resolve()
    if not provider_uri.exists():
        raise FileNotFoundError(f"Qlib provider does not exist: {provider_uri}")

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)

    calendar = D.calendar(start_time=args.start_time, end_time=args.end_time, freq="day")
    all_instruments = D.list_instruments(
        D.instruments("all"),
        start_time=args.start_time,
        end_time=args.end_time,
        as_list=True,
    )
    available = [inst for inst in args.instruments if inst in set(all_instruments)]
    if not available:
        available = all_instruments[:3]
    if not available:
        raise RuntimeError("No instruments are available in the requested date range.")

    features = D.features(
        available,
        DEFAULT_FIELDS,
        start_time=args.start_time,
        end_time=args.end_time,
        freq="day",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "smoke_test_sample.csv"
    report_path = args.output_dir / "smoke_test_report.json"
    features.reset_index().to_csv(sample_path, index=False)

    non_null = features.notna().sum().to_dict()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qlib_version": qlib.__version__,
        "provider_uri": str(provider_uri),
        "calendar_days": len(calendar),
        "instrument_count_in_window": len(all_instruments),
        "tested_instruments": available,
        "fields": DEFAULT_FIELDS,
        "rows": len(features),
        "non_null": {str(k): int(v) for k, v in non_null.items()},
        "sample_path": str(sample_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

