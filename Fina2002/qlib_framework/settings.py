from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_FRAMEWORK_DIR = Path(__file__).resolve().parent

PROVIDER_URI = Path(
    os.environ.get("QUANT_PROJECT_QLIB_PROVIDER_URI", QLIB_FRAMEWORK_DIR / "data" / "cn_data")
).expanduser()

RAW_DAILY_CSV = Path(
    os.environ.get("QUANT_ALPHAAGENT_DAILY_CSV", PROJECT_ROOT / "external_data" / "alphaagent_qlib_daily_ashare.csv")
).expanduser()
DUMP_INPUT_DIR = Path(
    os.environ.get("QUANT_QLIB_DUMP_INPUT_DIR", PROJECT_ROOT / "external_data" / "qlib_dump_input" / "raw_data")
).expanduser()
DUMP_SCRIPT = Path(
    os.environ.get("QUANT_QLIB_DUMP_SCRIPT", PROJECT_ROOT / "external_tools" / "dump_bin.py")
).expanduser()

REGION = "cn"
FREQ = "day"
DATE_FIELD = "date"
SYMBOL_FIELD = "code"

FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "return",
    "factor",
)

OUTPUT_DIR = QLIB_FRAMEWORK_DIR / "output"
