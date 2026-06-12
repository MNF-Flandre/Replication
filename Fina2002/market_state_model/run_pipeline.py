from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from market_state_model.config import MarketStateConfig
from market_state_model.discover_market_data import (
    discover_market_data,
    inventory_raw_files,
    write_missing_market_data_report,
)
from market_state_model.factors import build_state_factors
from market_state_model.filtering import forward_filter_probabilities, smoothed_probabilities_for_diagnostics
from market_state_model.hmm_model import fit_state_model
from market_state_model.preprocess import build_market_dataset
from market_state_model.signal import add_entry_score
from market_state_model.transitions import add_transition_features
from market_state_model.validation import (
    write_model_parameters,
    write_state_summary_report,
    write_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run market state Gaussian HMM pipeline.")
    parser.add_argument("--force-rebuild-stock-cache", action="store_true")
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--index-code", default=None)
    return parser.parse_args()


def run(config: MarketStateConfig) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "intermediate").mkdir(parents=True, exist_ok=True)

    audits = discover_market_data(config)
    write_missing_market_data_report(config, audits)
    inventory = inventory_raw_files(config)
    inventory.to_csv(config.output_dir / "source_inventory.csv", index=False, encoding="utf-8-sig")

    if any(item.status == "missing" for item in audits):
        missing = [item.name for item in audits if item.status == "missing"]
        raise RuntimeError(f"Core market data missing; aborting HMM result generation: {missing}")

    market = build_market_dataset(config)
    factors = build_state_factors(market, config)
    if len(factors) < 500:
        raise RuntimeError(f"Only {len(factors)} factor observations are available; not enough for stable HMM.")

    model = fit_state_model(factors, config)
    filtered = forward_filter_probabilities(model, factors)
    transitions = add_transition_features(filtered, config)
    scored = add_entry_score(transitions, config)

    merge_cols = [
        "date",
        "market_ret",
        "index_level",
        "realized_vol_20",
        "downside_vol_20",
        "cs_return_std",
        "avg_abs_stock_return",
        "advancer_ratio",
        "above_ma20_ratio",
        "above_ma60_ratio",
        "new_high_60_ratio",
        "new_low_60_ratio",
        "ret_count",
        "total_circulated_mktcap",
        "composite_trade_value",
        "composite_trade_shares",
        "composite_stock_count",
        "market_value",
        "market_circulated_mkt_value",
        "market_volume",
        "market_amount",
        "market_turnover_proxy",
        "margin_balance",
        "margin_buy",
        "margin_repay",
        "short_selling_value",
        "margin_total_trade",
        "margin_net_buy_ratio",
        "short_selling_value_ratio",
        "ma20",
        "ma60",
        "E",
        "D",
        "B",
        "Liq",
        "F",
        "data_quality_flags",
    ]
    final = scored.merge(factors[merge_cols], on="date", how="left")
    final["data_split"] = "test"
    final.loc[(final["date"] >= model.train_start) & (final["date"] <= model.train_end), "data_split"] = "train"

    required_order = [
        "date",
        "p_high_entropy",
        "p_low_bull",
        "p_low_bear",
        "posterior_entropy",
        "delta_p_low_bull",
        "delta_p_high_entropy",
        "delta_p_low_bear",
        "xi_H_to_Lplus",
        "xi_H_to_Lminus",
        "transition_score",
        "entry_score",
        "market_regime",
        "bullish_transition_signal",
        "bearish_transition_signal",
        "data_quality_flags",
    ]
    extra_cols = [col for col in final.columns if col not in required_order]
    final = final[required_order + extra_cols].sort_values("date").reset_index(drop=True)

    final.to_csv(config.output_dir / "market_state_probabilities.csv", index=False, encoding="utf-8-sig")
    final[
        [
            "date",
            "xi_H_to_Lplus",
            "xi_H_to_Lminus",
            "transition_score",
            "delta_p_low_bull",
            "delta_p_high_entropy",
            "delta_p_low_bear",
            "bullish_transition_signal",
            "bearish_transition_signal",
            "market_regime",
        ]
    ].to_csv(config.output_dir / "market_state_transitions.csv", index=False, encoding="utf-8-sig")
    final[
        [
            "date",
            "entry_score",
            "p_low_bull",
            "p_low_bear",
            "p_high_entropy",
            "posterior_entropy",
            "bullish_transition_signal",
            "bearish_transition_signal",
            "market_regime",
        ]
    ].to_csv(config.output_dir / "entry_score.csv", index=False, encoding="utf-8-sig")

    smoothed = smoothed_probabilities_for_diagnostics(model, factors)
    smoothed.to_csv(config.output_dir / "smoothed_probabilities_diagnostic_only.csv", index=False, encoding="utf-8-sig")
    write_model_parameters(config.output_dir, model)
    write_state_summary_report(config.output_dir, model, final, config)
    write_validation_report(config.output_dir, model, final, smoothed, config)
    return final


def main() -> None:
    args = parse_args()
    config = MarketStateConfig()
    config.force_rebuild_stock_cache = args.force_rebuild_stock_cache
    if args.train_end_date:
        config.train_end_date = args.train_end_date
    if args.index_code:
        config.market_index_code = args.index_code

    final = run(config)
    print(f"Wrote {len(final)} market-state observations to {config.output_dir}")
    print(f"Date range: {final['date'].min().date()} to {final['date'].max().date()}")
    print(f"Bullish transition signals: {int(final['bullish_transition_signal'].sum())}")
    print(f"Bearish transition signals: {int(final['bearish_transition_signal'].sum())}")


if __name__ == "__main__":
    main()
