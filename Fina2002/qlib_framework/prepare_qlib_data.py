from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import qlib

from qlib_framework.settings import (
    DATE_FIELD,
    DUMP_INPUT_DIR,
    DUMP_SCRIPT,
    FIELDS,
    FREQ,
    PROVIDER_URI,
    RAW_DAILY_CSV,
    SYMBOL_FIELD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build project-local Qlib binary data from the prepared AlphaAgent daily CSV split."
    )
    parser.add_argument("--raw-data-dir", type=Path, default=DUMP_INPUT_DIR)
    parser.add_argument("--provider-uri", type=Path, default=PROVIDER_URI)
    parser.add_argument("--dump-script", type=Path, default=DUMP_SCRIPT)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only dump the first N instruments for debugging.")
    parser.add_argument("--rebuild", action="store_true", help="Delete the target provider directory before dumping.")
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_dump_class(dump_script: Path):
    spec = importlib.util.spec_from_file_location("project_qlib_dump_bin", dump_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dump script: {dump_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ProcessPoolExecutor = module.ThreadPoolExecutor
    return module.DumpDataAll


def already_built(provider_uri: Path) -> bool:
    return (
        (provider_uri / "calendars" / f"{FREQ}.txt").exists()
        and (provider_uri / "instruments" / "all.txt").exists()
        and (provider_uri / "features").is_dir()
    )


def write_manifest(provider_uri: Path, raw_data_dir: Path, max_workers: int, limit: int | None) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qlib_version": qlib.__version__,
        "provider_uri": str(provider_uri.resolve()),
        "raw_daily_csv": str(RAW_DAILY_CSV),
        "raw_data_dir": str(raw_data_dir.resolve()),
        "dump_script": str(DUMP_SCRIPT),
        "fields": list(FIELDS),
        "date_field": DATE_FIELD,
        "symbol_field": SYMBOL_FIELD,
        "freq": FREQ,
        "max_workers": max_workers,
        "limit": limit,
    }
    (provider_uri / "project_qlib_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    raw_data_dir = args.raw_data_dir.expanduser().resolve()
    provider_uri = args.provider_uri.expanduser().resolve()
    dump_script = args.dump_script.expanduser().resolve()

    require_path(raw_data_dir, "raw data directory")
    require_path(dump_script, "dump script")

    if args.rebuild and provider_uri.exists():
        shutil.rmtree(provider_uri)

    if already_built(provider_uri) and not args.rebuild:
        print(f"Qlib provider already exists: {provider_uri}")
        write_manifest(provider_uri, raw_data_dir, args.max_workers, args.limit)
        return

    provider_uri.mkdir(parents=True, exist_ok=True)
    dump_cls = load_dump_class(dump_script)
    dumper = dump_cls(
        data_path=str(raw_data_dir),
        qlib_dir=str(provider_uri),
        freq=FREQ,
        max_workers=args.max_workers,
        date_field_name=DATE_FIELD,
        file_suffix=".csv",
        symbol_field_name=SYMBOL_FIELD,
        include_fields=",".join(FIELDS),
        limit_nums=args.limit,
    )
    dumper.dump()
    write_manifest(provider_uri, raw_data_dir, args.max_workers, args.limit)
    print(f"Qlib provider ready: {provider_uri}")


if __name__ == "__main__":
    main()
