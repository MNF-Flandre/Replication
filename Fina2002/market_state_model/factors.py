from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MarketStateConfig


FACTOR_COMPONENTS: dict[str, list[str]] = {
    "E": [
        "z_realized_vol_20",
        "z_downside_vol_20",
        "z_cs_return_std",
        "z_avg_abs_stock_return",
        "z_breadth_mixed",
    ],
    "D": [
        "z_market_ret",
        "z_momentum_20",
        "z_momentum_60",
        "z_ma20_distance",
        "z_ma60_distance",
    ],
    "B": [
        "z_advancer_ratio",
        "z_above_ma20_ratio",
        "z_above_ma60_ratio",
        "z_new_high_minus_low",
    ],
    "Liq": [
        "z_market_amount_growth_20",
        "z_market_turnover_proxy",
        "z_market_turnover_proxy_change_20",
        "z_neg_amihud_proxy",
        "z_market_cap_growth_20",
        "z_stock_coverage_change_20",
        "z_neg_avg_abs_stock_return",
    ],
    "F": [
        "z_margin_balance_growth_20",
        "z_margin_net_buy_ratio",
        "z_margin_trade_growth_20",
        "z_neg_short_selling_value_ratio",
        "z_neg_shibor_7d_change_20",
        "z_neg_shibor_term_spread",
        "z_neg_shibor_7d",
    ],
}


def rolling_zscore(series: pd.Series, config: MarketStateConfig) -> pd.Series:
    mean = series.rolling(config.zscore_window, min_periods=config.zscore_min_periods).mean()
    std = series.rolling(config.zscore_window, min_periods=config.zscore_min_periods).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(-config.winsor_z, config.winsor_z)


def add_raw_market_features(df: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    out = df.copy().sort_values("date")
    r = out["market_ret"]
    out["realized_vol_20"] = r.rolling(config.rolling_short, min_periods=15).std() * np.sqrt(config.annualization)
    downside_square = np.minimum(r, 0.0) ** 2
    out["downside_vol_20"] = np.sqrt(
        downside_square.rolling(config.rolling_short, min_periods=15).mean() * config.annualization
    )
    out["momentum_20"] = out["index_level"].pct_change(config.rolling_short)
    out["momentum_60"] = out["index_level"].pct_change(config.rolling_medium)
    out["ma20"] = out["index_level"].rolling(config.rolling_short, min_periods=15).mean()
    out["ma60"] = out["index_level"].rolling(config.rolling_medium, min_periods=40).mean()
    out["ma20_distance"] = out["index_level"] / out["ma20"] - 1.0
    out["ma60_distance"] = out["index_level"] / out["ma60"] - 1.0
    out["breadth_mixed"] = 1.0 - 2.0 * (out["advancer_ratio"] - 0.5).abs()
    out["new_high_minus_low"] = out["new_high_60_ratio"] - out["new_low_60_ratio"]
    market_cap_source = out["market_circulated_mkt_value"].where(
        out["market_circulated_mkt_value"].notna(), out["total_circulated_mktcap"]
    )
    safe_mktcap = market_cap_source.where(market_cap_source > 0)
    safe_ret_count = out["ret_count"].where(out["ret_count"] > 0)
    safe_amount = out["market_amount"].where(out["market_amount"] > 0)
    safe_turnover = out["market_turnover_proxy"].where(out["market_turnover_proxy"] > 0)
    safe_margin_balance = out["margin_balance"].where(out["margin_balance"] > 0)
    safe_margin_trade = out["margin_total_trade"].where(out["margin_total_trade"] > 0)

    out["market_cap_growth_20"] = np.log(safe_mktcap).diff(config.rolling_short)
    out["stock_coverage_change_20"] = np.log(safe_ret_count).diff(config.rolling_short)
    out["market_amount_growth_20"] = np.log(safe_amount).diff(config.rolling_short)
    out["market_turnover_proxy_change_20"] = np.log(safe_turnover).diff(config.rolling_short)
    out["amihud_proxy"] = out["market_ret"].abs() / safe_amount
    out["neg_amihud_proxy"] = -out["amihud_proxy"]
    out["neg_avg_abs_stock_return"] = -out["avg_abs_stock_return"]
    out["margin_balance_growth_20"] = np.log(safe_margin_balance).diff(config.rolling_short)
    out["margin_trade_growth_20"] = np.log(safe_margin_trade).diff(config.rolling_short)
    out["neg_short_selling_value_ratio"] = -out["short_selling_value_ratio"]
    out["shibor_7d_change_20"] = out["shibor_7d"].diff(config.rolling_short)
    out["shibor_term_spread"] = out["shibor_90d"] - out["shibor_1d"]
    out["neg_shibor_7d_change_20"] = -out["shibor_7d_change_20"]
    out["neg_shibor_term_spread"] = -out["shibor_term_spread"]
    out["neg_shibor_7d"] = -out["shibor_7d"]
    return out


def build_state_factors(market_df: pd.DataFrame, config: MarketStateConfig) -> pd.DataFrame:
    out = add_raw_market_features(market_df, config)
    raw_to_z = {
        "market_ret": "z_market_ret",
        "realized_vol_20": "z_realized_vol_20",
        "downside_vol_20": "z_downside_vol_20",
        "cs_return_std": "z_cs_return_std",
        "avg_abs_stock_return": "z_avg_abs_stock_return",
        "breadth_mixed": "z_breadth_mixed",
        "momentum_20": "z_momentum_20",
        "momentum_60": "z_momentum_60",
        "ma20_distance": "z_ma20_distance",
        "ma60_distance": "z_ma60_distance",
        "advancer_ratio": "z_advancer_ratio",
        "above_ma20_ratio": "z_above_ma20_ratio",
        "above_ma60_ratio": "z_above_ma60_ratio",
        "new_high_minus_low": "z_new_high_minus_low",
        "market_amount_growth_20": "z_market_amount_growth_20",
        "market_turnover_proxy": "z_market_turnover_proxy",
        "market_turnover_proxy_change_20": "z_market_turnover_proxy_change_20",
        "neg_amihud_proxy": "z_neg_amihud_proxy",
        "market_cap_growth_20": "z_market_cap_growth_20",
        "stock_coverage_change_20": "z_stock_coverage_change_20",
        "neg_avg_abs_stock_return": "z_neg_avg_abs_stock_return",
        "margin_balance_growth_20": "z_margin_balance_growth_20",
        "margin_net_buy_ratio": "z_margin_net_buy_ratio",
        "margin_trade_growth_20": "z_margin_trade_growth_20",
        "neg_short_selling_value_ratio": "z_neg_short_selling_value_ratio",
        "neg_shibor_7d_change_20": "z_neg_shibor_7d_change_20",
        "neg_shibor_term_spread": "z_neg_shibor_term_spread",
        "neg_shibor_7d": "z_neg_shibor_7d",
    }
    for raw, zcol in raw_to_z.items():
        out[zcol] = rolling_zscore(out[raw], config)

    for factor, components in FACTOR_COMPONENTS.items():
        out[factor] = out[components].mean(axis=1, skipna=True)

    required = ["date", *config.state_factor_columns]
    out = out.dropna(subset=required).copy()
    out["data_quality_flags"] = out["data_quality_flags"].fillna("")
    out = out.sort_values("date").reset_index(drop=True)
    config.factor_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.factor_path, index=False, encoding="utf-8-sig")
    return out
