from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import MarketStateConfig


TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, **kwargs)


def load_composite_market_returns(config: MarketStateConfig) -> pd.DataFrame:
    if not config.market_return_daily_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(config.market_return_daily_path)
    df = df.loc[df["Markettype"] == config.market_return_market_type].copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Trddt"], errors="coerce")
    return_candidates = ["Cdretwdos", "Cdretwdtl", "Cdretmdeq", "Cdretwdeq"]
    df["market_ret"] = np.nan
    for col in return_candidates:
        if col in df.columns:
            df["market_ret"] = df["market_ret"].fillna(pd.to_numeric(df[col], errors="coerce"))
    df["composite_trade_value"] = pd.to_numeric(df.get("Cnvaltrdtl"), errors="coerce")
    df["composite_trade_shares"] = pd.to_numeric(df.get("Cnshrtrdtl"), errors="coerce")
    df["composite_stock_count"] = pd.to_numeric(df.get("Cdnstkcal"), errors="coerce")
    out = df.dropna(subset=["date", "market_ret"]).sort_values("date")
    out = out[["date", "market_ret", "composite_trade_value", "composite_trade_shares", "composite_stock_count"]]
    out = out.drop_duplicates("date")
    out["index_code"] = f"TRD_Cndalym_{config.market_return_market_type}"
    out["index_level"] = (1.0 + out["market_ret"]).cumprod()
    return out.reset_index(drop=True)


def load_index_returns(config: MarketStateConfig) -> pd.DataFrame:
    composite = load_composite_market_returns(config)
    if not composite.empty:
        return composite

    if not config.index_daily_path.exists():
        raise FileNotFoundError(f"Index daily file not found: {config.index_daily_path}")
    df = pd.read_csv(config.index_daily_path, dtype={"Indexcd": str})
    df["date"] = pd.to_datetime(df["Trddt"], errors="coerce")
    df["market_ret"] = pd.to_numeric(df["Retindex"], errors="coerce")
    df = df.dropna(subset=["date", "market_ret"])

    available = set(df["Indexcd"].dropna().astype(str))
    index_code = next((code for code in (config.market_index_code, *config.fallback_index_codes) if code in available), None)
    if index_code is None:
        raise ValueError(f"No configured market index found. Available examples: {sorted(available)[:10]}")

    out = df.loc[df["Indexcd"] == index_code, ["date", "market_ret"]].copy()
    out = out.sort_values("date").drop_duplicates("date")
    out["index_code"] = index_code
    out["index_level"] = (1.0 + out["market_ret"]).cumprod()
    out["composite_trade_value"] = np.nan
    out["composite_trade_shares"] = np.nan
    out["composite_stock_count"] = np.nan
    return out.reset_index(drop=True)


def choose_stock_price_dir(config: MarketStateConfig) -> Path:
    for path in config.stock_price_dirs:
        if path.exists() and list(path.glob("TRD_BwardQuotationMonth*.csv")):
            return path
    raise FileNotFoundError("No stock price directory with TRD_BwardQuotationMonth*.csv was found.")


def _aggregate_one_stock_file(path: Path) -> pd.DataFrame:
    required = ["Symbol", "CloseDate", "ClosePrice"]
    optional = ["Filling", "CirculatedMarketValue"]
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = required + [col for col in optional if col in available]
    missing_required = [col for col in required if col not in available]
    if missing_required:
        raise ValueError(f"{path} missing required stock columns: {missing_required}")

    df = pd.read_csv(path, usecols=usecols, dtype={"Symbol": str})
    df["date"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    df["close"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
    if "CirculatedMarketValue" in df.columns:
        df["cmv"] = pd.to_numeric(df["CirculatedMarketValue"], errors="coerce")
    else:
        df["cmv"] = np.nan
    if "Filling" in df.columns:
        df["filling"] = pd.to_numeric(df["Filling"], errors="coerce")
    else:
        df["filling"] = 0
    df = df.dropna(subset=["Symbol", "date", "close"])
    df = df[(df["close"] > 0) & (df["filling"] == 0)]
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["Symbol", "date"]).drop_duplicates(["Symbol", "date"], keep="last")
    grouped = df.groupby("Symbol", sort=False)
    df["stock_ret"] = grouped["close"].pct_change()
    df.loc[df["stock_ret"].abs() > 3.0, "stock_ret"] = np.nan

    ma20 = grouped["close"].rolling(20, min_periods=15).mean().reset_index(level=0, drop=True)
    ma60 = grouped["close"].rolling(60, min_periods=40).mean().reset_index(level=0, drop=True)
    high60 = grouped["close"].rolling(60, min_periods=40).max().reset_index(level=0, drop=True)
    low60 = grouped["close"].rolling(60, min_periods=40).min().reset_index(level=0, drop=True)
    df["above_ma20"] = (df["close"] >= ma20).where(ma20.notna())
    df["above_ma60"] = (df["close"] >= ma60).where(ma60.notna())
    df["new_high_60"] = (df["close"] >= high60).where(high60.notna())
    df["new_low_60"] = (df["close"] <= low60).where(low60.notna())

    valid_ret = df["stock_ret"].notna()
    df["ret_sum"] = df["stock_ret"].where(valid_ret, 0.0)
    df["ret_sq_sum"] = (df["stock_ret"] ** 2).where(valid_ret, 0.0)
    df["abs_ret_sum"] = df["stock_ret"].abs().where(valid_ret, 0.0)
    df["ret_count"] = valid_ret.astype(int)
    df["advancer_count"] = (df["stock_ret"] > 0).astype(int)
    df["decliner_count"] = (df["stock_ret"] < 0).astype(int)
    df["cmv_available"] = df["cmv"].notna().astype(int)

    for col in ["above_ma20", "above_ma60", "new_high_60", "new_low_60"]:
        df[f"{col}_count"] = (df[col] == True).astype(int)
        df[f"{col}_valid"] = df[col].notna().astype(int)

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            stock_count=("Symbol", "nunique"),
            total_circulated_mktcap=("cmv", "sum"),
            cmv_available=("cmv_available", "sum"),
            ret_sum=("ret_sum", "sum"),
            ret_sq_sum=("ret_sq_sum", "sum"),
            abs_ret_sum=("abs_ret_sum", "sum"),
            ret_count=("ret_count", "sum"),
            advancer_count=("advancer_count", "sum"),
            decliner_count=("decliner_count", "sum"),
            above_ma20_count=("above_ma20_count", "sum"),
            above_ma20_valid=("above_ma20_valid", "sum"),
            above_ma60_count=("above_ma60_count", "sum"),
            above_ma60_valid=("above_ma60_valid", "sum"),
            new_high_60_count=("new_high_60_count", "sum"),
            new_high_60_valid=("new_high_60_valid", "sum"),
            new_low_60_count=("new_low_60_count", "sum"),
            new_low_60_valid=("new_low_60_valid", "sum"),
        )
        .sort_values("date")
    )
    return daily


def aggregate_stock_daily_features(config: MarketStateConfig) -> pd.DataFrame:
    cache_path = config.stock_cache_path
    if cache_path.exists() and not config.force_rebuild_stock_cache:
        out = pd.read_csv(cache_path, parse_dates=["date"])
        return out.sort_values("date").reset_index(drop=True)

    stock_dir = choose_stock_price_dir(config)
    files = sorted(stock_dir.glob("TRD_BwardQuotationMonth*.csv"))
    if not files:
        raise FileNotFoundError(f"No stock price csv files found in {stock_dir}")

    daily_parts = []
    for path in files:
        part = _aggregate_one_stock_file(path)
        if not part.empty:
            daily_parts.append(part)
    if not daily_parts:
        raise ValueError("Stock price files were found, but no usable rows remained after filtering.")

    combined = pd.concat(daily_parts, ignore_index=True)
    sum_cols = [col for col in combined.columns if col != "date"]
    daily = combined.groupby("date", as_index=False)[sum_cols].sum().sort_values("date")
    daily.loc[daily["cmv_available"] <= 0, "total_circulated_mktcap"] = np.nan
    n = daily["ret_count"].replace(0, np.nan)
    daily["cs_return_mean"] = daily["ret_sum"] / n
    variance = (daily["ret_sq_sum"] - (daily["ret_sum"] ** 2) / n) / (n - 1)
    daily["cs_return_std"] = np.sqrt(variance.clip(lower=0))
    daily["avg_abs_stock_return"] = daily["abs_ret_sum"] / n
    daily["advancer_ratio"] = daily["advancer_count"] / n
    daily["decliner_ratio"] = daily["decliner_count"] / n

    for col in ["above_ma20", "above_ma60", "new_high_60", "new_low_60"]:
        valid = daily[f"{col}_valid"].replace(0, np.nan)
        daily[f"{col}_ratio"] = daily[f"{col}_count"] / valid

    keep = [
        "date",
        "stock_count",
        "ret_count",
        "total_circulated_mktcap",
        "cs_return_mean",
        "cs_return_std",
        "avg_abs_stock_return",
        "advancer_ratio",
        "decliner_ratio",
        "above_ma20_ratio",
        "above_ma60_ratio",
        "new_high_60_ratio",
        "new_low_60_ratio",
    ]
    out = daily[keep].sort_values("date").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return out


def load_shibor(config: MarketStateConfig) -> pd.DataFrame:
    if not config.shibor_path.exists():
        return pd.DataFrame(columns=["date", "shibor_1d", "shibor_7d", "shibor_30d", "shibor_90d"])
    df = pd.read_csv(config.shibor_path)
    df["date"] = pd.to_datetime(df["SgnDate"], errors="coerce")
    df["Shibor"] = pd.to_numeric(df["Shibor"], errors="coerce")
    pivot = df.pivot_table(index="date", columns="Term", values="Shibor", aggfunc="last").sort_index()
    term_map = {"1天": "shibor_1d", "7天": "shibor_7d", "30天": "shibor_30d", "90天": "shibor_90d"}
    out = pd.DataFrame(index=pivot.index)
    for term, name in term_map.items():
        out[name] = pivot[term] if term in pivot.columns else np.nan
    out = out.reset_index().sort_values("date")
    return out


def load_market_size_daily(config: MarketStateConfig) -> pd.DataFrame:
    if not config.market_size_daily_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(config.market_size_daily_path)
    df = df.loc[df["DataSgnCode"] == config.market_size_data_sgn_code].copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["SgnDate"], errors="coerce")
    numeric_cols = [
        "ListedCoNum",
        "StockNum",
        "MarketValue",
        "CirculatedMktValue",
        "Volume",
        "Amount",
        "AvgPE",
        "TurnoverRate1",
        "TurnoverRate2",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = df.rename(
        columns={
            "ListedCoNum": "market_listed_company_count",
            "StockNum": "market_stock_count",
            "MarketValue": "market_value",
            "CirculatedMktValue": "market_circulated_mkt_value",
            "Volume": "market_volume",
            "Amount": "market_amount",
            "AvgPE": "market_avg_pe",
            "TurnoverRate1": "market_turnover_rate_total",
            "TurnoverRate2": "market_turnover_rate_float",
        }
    )
    out["market_turnover_proxy"] = out["market_amount"] / out["market_circulated_mkt_value"].replace(0, np.nan)
    keep = [
        "date",
        "market_listed_company_count",
        "market_stock_count",
        "market_value",
        "market_circulated_mkt_value",
        "market_volume",
        "market_amount",
        "market_avg_pe",
        "market_turnover_rate_total",
        "market_turnover_rate_float",
        "market_turnover_proxy",
    ]
    return out.dropna(subset=["date"]).sort_values("date")[keep].drop_duplicates("date").reset_index(drop=True)


def load_margin_trading(config: MarketStateConfig) -> pd.DataFrame:
    if not config.margin_trading_path.exists():
        return pd.DataFrame()
    df = read_csv_with_fallback(config.margin_trading_path)
    if df.empty:
        return pd.DataFrame()
    cols = df.columns.tolist()
    rename = {
        cols[1]: "date",
        cols[3]: "stat_period",
        cols[4]: "margin_balance",
        cols[5]: "margin_buy",
        cols[6]: "margin_repay",
        cols[8]: "short_selling_value",
        cols[10]: "margin_total_trade",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stat_period"] = pd.to_numeric(df["stat_period"], errors="coerce")
    df = df.loc[df["stat_period"] == 5].copy()
    for col in ["margin_balance", "margin_buy", "margin_repay", "short_selling_value", "margin_total_trade"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = (
        df.groupby("date", as_index=False)
        .agg(
            margin_balance=("margin_balance", "sum"),
            margin_buy=("margin_buy", "sum"),
            margin_repay=("margin_repay", "sum"),
            short_selling_value=("short_selling_value", "sum"),
            margin_total_trade=("margin_total_trade", "sum"),
            margin_market_count=("stat_period", "count"),
        )
        .sort_values("date")
    )
    denom = out["margin_balance"].replace(0, np.nan)
    out["margin_net_buy_ratio"] = (out["margin_buy"] - out["margin_repay"]) / denom
    out["short_selling_value_ratio"] = out["short_selling_value"] / (
        out["margin_balance"] + out["short_selling_value"]
    ).replace(0, np.nan)
    return out.reset_index(drop=True)


def build_market_dataset(config: MarketStateConfig) -> pd.DataFrame:
    index_df = load_index_returns(config)
    stock_df = aggregate_stock_daily_features(config)
    shibor_df = load_shibor(config)
    market_size_df = load_market_size_daily(config)
    margin_df = load_margin_trading(config)

    out = index_df.merge(stock_df, on="date", how="left")
    if not market_size_df.empty:
        out = out.merge(market_size_df, on="date", how="left")
    else:
        for col in [
            "market_listed_company_count",
            "market_stock_count",
            "market_value",
            "market_circulated_mkt_value",
            "market_volume",
            "market_amount",
            "market_avg_pe",
            "market_turnover_rate_total",
            "market_turnover_rate_float",
            "market_turnover_proxy",
        ]:
            out[col] = np.nan

    if not margin_df.empty:
        out = pd.merge_asof(out.sort_values("date"), margin_df.sort_values("date"), on="date", direction="backward")
    else:
        for col in [
            "margin_balance",
            "margin_buy",
            "margin_repay",
            "short_selling_value",
            "margin_total_trade",
            "margin_market_count",
            "margin_net_buy_ratio",
            "short_selling_value_ratio",
        ]:
            out[col] = np.nan

    if not shibor_df.empty:
        out = pd.merge_asof(out.sort_values("date"), shibor_df.sort_values("date"), on="date", direction="backward")
    else:
        for col in ["shibor_1d", "shibor_7d", "shibor_30d", "shibor_90d"]:
            out[col] = np.nan

    flags = []
    for _, row in out.iterrows():
        row_flags = []
        if pd.isna(row.get("ret_count")) or row.get("ret_count", 0) < config.min_stock_return_count:
            row_flags.append("low_stock_breadth_coverage")
        if pd.isna(row.get("market_amount")):
            row_flags.append("missing_market_amount")
        if pd.isna(row.get("market_circulated_mkt_value")) and pd.isna(row.get("total_circulated_mktcap")):
            row_flags.append("missing_market_cap")
        if pd.isna(row.get("margin_balance")):
            row_flags.append("missing_margin_trading")
        if pd.isna(row.get("shibor_7d")):
            row_flags.append("missing_shibor")
        flags.append(";".join(row_flags) if row_flags else "ok")
    out["data_quality_flags"] = flags
    return out.sort_values("date").reset_index(drop=True)
