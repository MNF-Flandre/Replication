# Core Reproducibility Pipeline

This package contains source code for the core Gap x Market State pipeline. 

## Pipeline

### 1. Prepare market data

Convert raw daily stock data into a Qlib provider directory:

```powershell
python qlib_framework/prepare_qlib_data.py `
  --raw-data-dir <raw_daily_csv_dir> `
  --provider-uri <qlib_provider_uri>
```

Build index return and constituent helper files from RESSET index constituent data:

```powershell
python etf_weight/build_resset_index_returns.py
```

Expected outputs from this step are index daily returns and constituent interval files. They are inputs to the stock selection and strategy scripts.

### 2. Build optimal leverage and Gap

Construct firm-level optimal leverage `d*` and leverage Gap:

```powershell
python optimal_leverage_model/run_pipeline.py `
  --source-root <raw_financial_statement_root> `
  --output-dir <gap_output_dir>
```

Optional validation and alternative proxy construction:

```powershell
python optimal_leverage_model/validate_optimal_leverage.py
python optimal_leverage_model/generate_consistent_finance_proxy_variant.py
```

The main downstream input is a Gap result table with firm id, reporting period, observed leverage, optimal leverage, and leverage gap fields.

### 3. Build market-state factors

Create market-state factor data using the market state model pipeline:

```powershell
python market_state_model/run_pipeline.py
```

The HMM scripts expect a factor table such as:

```text
market_state_model/output/intermediate/state_factors.csv
```

### 4. Train the HMM market-state recognizer

The locked state recognizer is a 4-state expanding-window diagonal Gaussian HMM with forward filtering.

```powershell
python market_state_model/run_feature_tuned_expanding_hmm.py `
  --factor-path <state_factor_csv> `
  --output-dir <hmm_output_dir> `
  --n-states 4 `
  --refit-freq Q `
  --initial-train-end 2014-12-31 `
  --hmm-max-iter 120
```

The Rust acceleration library is included under `qlib_framework/_rust/` and `rust_ext/quant_rust_core/`. If the DLL is available, HMM EM fitting and rolling max drawdown calculation use Rust automatically. To force the Python fallback:

```powershell
$env:QLIB_RUST_ACCEL = "0"
```

### 5. Integrate Gap and market state

Build point-in-time integrated panels:

```powershell
python integrated_gap_market_state/run_pit_integration.py
```

This aligns accounting report availability, Gap measures, market-state probabilities, and rebalance dates.

### 6. Run the core CS500 Gap x HMM strategy

Run the locked core strategy:

```powershell
python qlib_framework/run_core_cs500_gap_state_strategy.py `
  --start-date 2015-01-01 `
  --end-date 2025-12-31 `
  --gap-path <gap_result_csv> `
  --hmm-path <hmm_probability_csv> `
  --component-path <index_component_intervals_csv> `
  --benchmark-path <index_daily_returns_csv> `
  --market-daily-root <stock_daily_parquet_root> `
  --output-dir <strategy_output_dir>
```

Run robustness checks:

```powershell
python qlib_framework/run_core_cs500_gap_state_robustness.py --output-dir <robustness_output_dir>
python qlib_framework/run_core_cs500_time_hmm_robustness.py --output-dir <time_robustness_output_dir>
```

## Notes

- Data paths in scripts may need to be changed to local paths before running on another machine.
- The package intentionally excludes data files, output folders, plots, rendered documents, and Python cache files.
- The core locked convention is: no smoothed HMM probabilities, use forward-filtered HMM output, and do not recompute `d*` inside the strategy runner.

