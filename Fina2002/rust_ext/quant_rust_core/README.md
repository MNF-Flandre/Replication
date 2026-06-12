# quant_rust_core

Rust acceleration library for selected compute-heavy routines in the quant project.

Current exported function:

- `rolling_max_drawdown_grouped`: grouped fixed-window rolling max drawdown for `prev_quarter_max_drawdown`.
- `fit_diagonal_gaussian_hmm_seeded`: diagonal Gaussian HMM EM fitting with Python-provided initial labels.

Build on Windows:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
cargo build --release
Copy-Item target\release\quant_rust_core.dll ..\..\qlib_framework\_rust\quant_rust_core.dll -Force
```

Python integration lives in `qlib_framework/rust_accel.py`.

Set `QLIB_RUST_ACCEL=0` to force the Python/pandas fallback for both rolling max drawdown and HMM fitting.
