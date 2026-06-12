from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def env_path_list(name: str, default: list[Path]) -> list[Path]:
    value = os.environ.get(name)
    if not value:
        return default
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


@dataclass
class MarketStateConfig:
    project_dir: Path = Path(__file__).resolve().parents[1]
    module_dir: Path = Path(__file__).resolve().parent
    output_dir: Path = Path(__file__).resolve().parent / "output"
    local_raw_data_dir: Path = Path(__file__).resolve().parent / "raw_data"

    raw_data_roots: list[Path] = field(
        default_factory=lambda: [
            Path(__file__).resolve().parent / "raw_data",
            env_path("QUANT_RAWDATA_ROOT", Path(__file__).resolve().parents[1] / "external_data"),
            env_path("QUANT_STOCK_PRICE_ROOT", Path(__file__).resolve().parents[1] / "external_data" / "stk_price"),
        ]
    )
    index_daily_path: Path = env_path(
        "QUANT_INDEX_DAILY_PATH",
        Path(__file__).resolve().parents[1] / "external_data" / "index" / "CAPMR_Idxdalyr.csv",
    )
    stock_price_dirs: list[Path] = field(
        default_factory=lambda: env_path_list(
            "QUANT_STOCK_PRICE_DIRS",
            [
                Path(__file__).resolve().parents[1] / "external_data" / "stk_price",
                Path(__file__).resolve().parent / "raw_data" / "stk_price",
            ],
        )
    )
    shibor_path: Path = env_path(
        "QUANT_SHIBOR_PATH",
        Path(__file__).resolve().parents[1] / "external_data" / "independent_variable" / "Alpha" / "MBK_SHIBORM.csv",
    )
    margin_trading_path: Path = Path(__file__).resolve().parent / "raw_data" / "margin_trading" / "RESSET_MTMARSSTRDSTAT_1.csv"
    market_size_daily_path: Path = Path(__file__).resolve().parent / "raw_data" / "market_size_daily" / "TRD_MKStructD.csv"
    market_return_daily_path: Path = Path(__file__).resolve().parent / "raw_data" / "market_return_daily" / "TRD_Cndalym.csv"
    market_return_market_type: int = 117
    market_size_data_sgn_code: int = 14

    market_index_code: str = "000300"
    fallback_index_codes: tuple[str, ...] = (
        "000300",
        "000001",
        "399001",
        "000902",
        "000903",
        "399903",
    )

    stock_chunksize: int = 500_000
    force_rebuild_stock_cache: bool = False
    min_stock_return_count: int = 300

    rolling_short: int = 20
    rolling_medium: int = 60
    rolling_long: int = 120
    annualization: int = 252
    zscore_window: int = 252
    zscore_min_periods: int = 120
    winsor_z: float = 5.0

    state_factor_columns: tuple[str, ...] = ("E", "D", "B", "Liq", "F")
    train_end_date: str = "2018-12-31"
    train_fraction_if_needed: float = 0.70

    hmm_n_states: int = 3
    hmm_max_iter: int = 200
    hmm_tol: float = 1e-5
    hmm_random_seed: int = 20260529
    covariance_floor: float = 1e-4
    transmat_prior: float = 1e-2
    startprob_prior: float = 1e-2

    bull_prob_min: float = 0.45
    bull_delta_min: float = 0.08
    entropy_delta_max: float = -0.02
    bear_delta_max: float = 0.04
    bear_prob_min: float = 0.45
    bear_delta_min: float = 0.08
    bull_delta_max: float = 0.04

    entry_gamma: float = 0.60
    entry_rho: float = 0.90
    entry_xi: float = 0.55
    entry_omega: float = 0.25

    entry_quantile_for_validation: float = 0.80
    golden_cross_fast_window: int = 20
    golden_cross_slow_window: int = 60
    golden_cross_lead_window: int = 20

    @property
    def stock_cache_path(self) -> Path:
        return self.output_dir / "intermediate" / "stock_daily_breadth_cache.csv"

    @property
    def factor_path(self) -> Path:
        return self.output_dir / "intermediate" / "state_factors.csv"
