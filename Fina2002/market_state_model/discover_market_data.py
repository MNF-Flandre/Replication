from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import MarketStateConfig


@dataclass
class SourceAudit:
    name: str
    status: str
    path: str
    fields: str
    use: str


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _first_existing_dir(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_dir():
            return path
    return None


def discover_market_data(config: MarketStateConfig) -> list[SourceAudit]:
    stock_dir = _first_existing_dir(config.stock_price_dirs)
    stock_files = sorted(stock_dir.glob("TRD_BwardQuotationMonth*.csv")) if stock_dir else []
    audits = [
        SourceAudit(
            name="market_index_returns",
            status="available" if _exists(config.market_return_daily_path) or _exists(config.index_daily_path) else "missing",
            path=str(config.market_return_daily_path if _exists(config.market_return_daily_path) else config.index_daily_path),
            fields="Markettype, Trddt, Cdretwdos/Cdretwdtl or Indexcd, Trddt, Retindex",
            use="D factor, realized volatility, downside volatility, trend and validation returns",
        ),
        SourceAudit(
            name="stock_price_breadth",
            status="available" if stock_files else "missing",
            path=str(stock_dir) if stock_dir else "",
            fields="Symbol, CloseDate, Filling, ClosePrice, CirculatedMarketValue",
            use="cross-sectional dispersion, advancing ratio, MA breadth, new-high/new-low breadth, market-cap proxy",
        ),
        SourceAudit(
            name="shibor_funding_proxy",
            status="available" if _exists(config.shibor_path) else "missing",
            path=str(config.shibor_path),
            fields="SgnDate, Term, Shibor",
            use="F factor proxy for funding pressure and risk appetite",
        ),
        SourceAudit(
            name="market_size_liquidity",
            status="available" if _exists(config.market_size_daily_path) else "missing",
            path=str(config.market_size_daily_path),
            fields="SgnDate, Amount, Volume, MarketValue, CirculatedMktValue",
            use="Liq factor: market turnover, trading amount, liquidity proxy",
        ),
        SourceAudit(
            name="margin_trading_risk_appetite",
            status="available" if _exists(config.margin_trading_path) else "missing",
            path=str(config.margin_trading_path),
            fields="TrdDt, MarBal, MarBuySum, MarRefdSum, SSSum, MarTotTrdSum",
            use="F factor: margin balance growth, margin net buying, short-selling pressure",
        ),
    ]
    return audits


def inventory_raw_files(config: MarketStateConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in config.raw_data_roots:
        if not root.exists():
            rows.append({"root": str(root), "path": "", "size_mb": None, "status": "missing_root"})
            continue
        for path in root.rglob("*"):
            if path.is_file():
                rows.append(
                    {
                        "root": str(root),
                        "path": str(path),
                        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                        "status": "available",
                    }
                )
    return pd.DataFrame(rows)


def write_missing_market_data_report(config: MarketStateConfig, audits: list[SourceAudit]) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "missing_market_data_report.md"
    missing_items = [
        (
            "官方换手率字段",
            "Liq_t",
            "补充的日市场规模表提供成交额、成交量和市值；原始 TurnoverRate 字段为空，因此当前用 Amount / CirculatedMktValue 构造成交活跃度代理。",
        ),
        (
            "上涨家数/下跌家数官方表",
            "B_t",
            "未发现直接的市场广度表；第一版从个股收盘价计算上涨比例、均线以上比例和新高/新低比例。",
        ),
        (
            "信用利差",
            "F_t",
            "未在当前扫描路径发现；当前 F_t 已接入融资融券和 SHIBOR，后续若补充企业债/国债利差，可继续增强。",
        ),
        (
            "价格冲击或买卖价差",
            "Liq_t",
            "未发现逐笔或盘口数据，无法构造严格价差；当前用 |return| / Amount 构造低价格冲击代理。",
        ),
    ]

    lines = [
        "# 市场状态 HMM 缺失数据报告",
        "",
        "## 已审计数据源",
        "",
        "| 数据源 | 状态 | 路径 | 可用字段 | 用途 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in audits:
        lines.append(f"| {item.name} | {item.status} | `{item.path}` | {item.fields} | {item.use} |")

    lines.extend(
        [
            "",
            "## 缺失或只能代理的市场变量",
            "",
            "| 缺失变量 | 影响因子 | 处理方式 |",
            "| --- | --- | --- |",
        ]
    )
    for variable, factor, treatment in missing_items:
        lines.append(f"| {variable} | {factor} | {treatment} |")

    usable = all(item.status == "available" for item in audits)
    lines.extend(
        [
            "",
            "## 第一版可行性判断",
            "",
            (
                "当前数据足以构造增强版可审计 Gaussian HMM：E、D、B 可由综合市场收益和个股日收盘价构造；"
                "Liq 已接入日市场规模的成交额/成交量/市值，F 已接入融资融券与 SHIBOR。"
                if usable
                else "当前核心数据源不完整；不应生成市场状态结果，应先补齐上表中状态为 missing 的数据。"
            ),
            "",
            "本报告只审计 HMM 辅助模块数据，不修改 `optimal_leverage_model`、`d^*` 或 `Gap`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
