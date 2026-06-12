from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import numpy as np


_LIB_NAME = "quant_rust_core"
_MODULE_DIR = Path(__file__).resolve().parent
_DLL_DIR = _MODULE_DIR / "_rust"


def _library_candidates() -> list[Path]:
    system = platform.system().lower()
    if system == "windows":
        names = [f"{_LIB_NAME}.dll"]
    elif system == "darwin":
        names = [f"lib{_LIB_NAME}.dylib"]
    else:
        names = [f"lib{_LIB_NAME}.so"]
    return [_DLL_DIR / name for name in names]


def _load_library() -> ctypes.CDLL | None:
    disabled = os.environ.get("QLIB_RUST_ACCEL", "").strip().lower()
    if disabled in {"0", "false", "no", "off"}:
        return None
    for path in _library_candidates():
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
            except OSError:
                continue
            func = lib.rolling_max_drawdown_grouped
            func.argtypes = [
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
            ]
            func.restype = ctypes.c_int

            hmm_func = lib.fit_diagonal_gaussian_hmm_seeded
            hmm_func.argtypes = [
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            hmm_func.restype = ctypes.c_int
            return lib
    return None


_LIB = _load_library()


def rust_available() -> bool:
    return _LIB is not None


def rolling_max_drawdown_grouped(
    returns: np.ndarray,
    group_ids: np.ndarray,
    window: int,
    min_periods: int,
) -> np.ndarray:
    if _LIB is None:
        raise RuntimeError("Rust acceleration library is not available.")
    returns_arr = np.ascontiguousarray(returns, dtype=np.float64)
    group_arr = np.ascontiguousarray(group_ids, dtype=np.int64)
    if returns_arr.ndim != 1 or group_arr.ndim != 1:
        raise ValueError("returns and group_ids must be one-dimensional arrays.")
    if returns_arr.shape[0] != group_arr.shape[0]:
        raise ValueError("returns and group_ids must have the same length.")
    out = np.empty(returns_arr.shape[0], dtype=np.float64)
    status = _LIB.rolling_max_drawdown_grouped(
        ctypes.c_size_t(returns_arr.shape[0]),
        returns_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        group_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_size_t(window),
        ctypes.c_size_t(min_periods),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise RuntimeError(f"Rust rolling_max_drawdown_grouped failed with status {status}.")
    return out


def fit_diagonal_gaussian_hmm_seeded(
    x: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    max_iter: int,
    tol: float,
    start_prior: float,
    trans_prior: float,
    covariance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    if _LIB is None:
        raise RuntimeError("Rust acceleration library is not available.")
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    label_arr = np.ascontiguousarray(labels, dtype=np.int64)
    if x_arr.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    if label_arr.ndim != 1 or label_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("labels must be one-dimensional and match x rows.")
    n, n_features = x_arr.shape
    start = np.empty(n_states, dtype=np.float64)
    trans = np.empty((n_states, n_states), dtype=np.float64)
    means = np.empty((n_states, n_features), dtype=np.float64)
    covars = np.empty((n_states, n_features), dtype=np.float64)
    monitor = np.empty(max_iter, dtype=np.float64)
    iterations = ctypes.c_size_t(0)

    status = _LIB.fit_diagonal_gaussian_hmm_seeded(
        ctypes.c_size_t(n),
        ctypes.c_size_t(n_states),
        ctypes.c_size_t(n_features),
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        label_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_size_t(max_iter),
        ctypes.c_double(tol),
        ctypes.c_double(start_prior),
        ctypes.c_double(trans_prior),
        ctypes.c_double(covariance_floor),
        start.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        trans.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        means.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        covars.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        monitor.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(iterations),
    )
    if status != 0:
        raise RuntimeError(f"Rust fit_diagonal_gaussian_hmm_seeded failed with status {status}.")
    return start, trans, means, covars, monitor[: iterations.value].astype(float).tolist()
