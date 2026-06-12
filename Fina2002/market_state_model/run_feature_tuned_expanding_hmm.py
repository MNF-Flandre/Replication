from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
module_dir_text = str(MODULE_DIR)
if module_dir_text in sys.path:
    sys.path.remove(module_dir_text)
project_dir_text = str(PROJECT_DIR)
if project_dir_text not in sys.path:
    sys.path.insert(0, project_dir_text)

import pandas as pd

from market_state_model.config import MarketStateConfig
from market_state_model.run_feature_tuned_direct_hmm import (
    FACTOR_PATH,
    TUNED_FEATURE_COLUMNS,
    add_signals,
    add_tuned_features,
    build_final_frame,
    fit_best_model,
    forward_filter,
    name_states,
)


OUT_DIR = MODULE_DIR / "output" / "feature_tuned_expanding_hmm"
FREQ_LABELS = {
    "M": "monthly",
    "Q": "quarterly",
    "A": "annual",
    "Y": "annual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature-tuned HMM with expanding-window refits before each evaluation node."
    )
    parser.add_argument("--factor-path", type=Path, default=FACTOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-states", type=int, default=4, choices=[3, 4])
    parser.add_argument("--refit-freq", default="Q", choices=["M", "Q", "A", "Y"])
    parser.add_argument("--initial-train-end", default="2018-12-31")
    parser.add_argument("--min-train-obs", type=int, default=750)
    parser.add_argument("--hmm-max-iter", type=int, default=120)
    return parser.parse_args()


def normalize_freq(freq: str) -> str:
    freq = str(freq).upper()
    return "A" if freq == "Y" else freq


def build_segments(factors: pd.DataFrame, initial_train_end: pd.Timestamp, freq: str) -> list[pd.DataFrame]:
    eval_frame = factors.loc[factors["date"] > initial_train_end].copy()
    if eval_frame.empty:
        return []
    periods = eval_frame["date"].dt.to_period(freq)
    return [part.copy() for _, part in eval_frame.groupby(periods, sort=True)]


def run_expanding(args: argparse.Namespace) -> dict[str, object]:
    config = MarketStateConfig()
    config.hmm_n_states = args.n_states
    config.hmm_max_iter = args.hmm_max_iter

    factors = pd.read_csv(args.factor_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    factors = add_tuned_features(factors)
    factors = factors.dropna(subset=list(TUNED_FEATURE_COLUMNS)).reset_index(drop=True)
    if factors.empty:
        raise ValueError("No usable factor rows after feature construction.")

    initial_target = pd.Timestamp(args.initial_train_end)
    train_dates = factors.loc[factors["date"] <= initial_target, "date"]
    if train_dates.empty:
        raise ValueError(f"No factor rows on or before initial train end: {args.initial_train_end}")
    initial_train_end = pd.Timestamp(train_dates.max())

    freq = normalize_freq(args.refit_freq)
    segments = build_segments(factors, initial_train_end, freq)
    if not segments:
        raise ValueError("No evaluation segments after the initial training window.")

    final_parts = []
    refit_rows = []
    init_parts = []
    state_parts = []

    for refit_id, segment in enumerate(segments, start=1):
        segment_start = pd.Timestamp(segment["date"].min())
        segment_end = pd.Timestamp(segment["date"].max())
        train = factors.loc[factors["date"] < segment_start].copy()
        if len(train) < args.min_train_obs:
            refit_rows.append(
                {
                    "refit_id": refit_id,
                    "segment_start": segment_start.date().isoformat(),
                    "segment_end": segment_end.date().isoformat(),
                    "train_start": None,
                    "train_end": None,
                    "train_obs": len(train),
                    "selected_initialization": None,
                    "status": "skipped_min_train_obs",
                }
            )
            continue

        model, selected_init, init_table = fit_best_model(train, config, args.n_states)
        raw_to_label, state_table = name_states(model, args.n_states)

        filter_frame = pd.concat([train, segment], ignore_index=True).sort_values("date").reset_index(drop=True)
        filtered = forward_filter(model, filter_frame, raw_to_label)
        scored = add_signals(filtered, config)
        final_all = build_final_frame(scored, filter_frame, train)
        final = final_all.loc[final_all["date"].between(segment_start, segment_end)].copy()
        final["data_split"] = "expanding_oos"
        final["rolling_spec"] = FREQ_LABELS[freq]
        final["refit_id"] = refit_id
        final["train_start"] = pd.Timestamp(train["date"].min()).date().isoformat()
        final["train_end"] = pd.Timestamp(train["date"].max()).date().isoformat()
        final["segment_start"] = segment_start.date().isoformat()
        final["segment_end"] = segment_end.date().isoformat()
        final["selected_initialization"] = selected_init
        final_parts.append(final)

        init_part = init_table.copy()
        init_part["refit_id"] = refit_id
        init_part["segment_start"] = segment_start.date().isoformat()
        init_part["segment_end"] = segment_end.date().isoformat()
        init_parts.append(init_part)

        state_part = state_table.copy()
        state_part["refit_id"] = refit_id
        state_part["segment_start"] = segment_start.date().isoformat()
        state_part["segment_end"] = segment_end.date().isoformat()
        state_parts.append(state_part)

        refit_rows.append(
            {
                "refit_id": refit_id,
                "segment_start": segment_start.date().isoformat(),
                "segment_end": segment_end.date().isoformat(),
                "train_start": pd.Timestamp(train["date"].min()).date().isoformat(),
                "train_end": pd.Timestamp(train["date"].max()).date().isoformat(),
                "train_obs": len(train),
                "segment_obs": len(segment),
                "selected_initialization": selected_init,
                "best_train_loglik": float(init_table["train_loglik"].max()),
                "iterations": int(init_table.loc[init_table["initialization"].eq(selected_init), "iterations"].iloc[0]),
                "status": "ok",
            }
        )
        print(
            f"[{refit_id:03d}/{len(segments):03d}] {segment_start.date()} to {segment_end.date()} "
            f"train_end={pd.Timestamp(train['date'].max()).date()} init={selected_init}"
        )

    if not final_parts:
        raise RuntimeError("No expanding-window HMM segment was fitted.")

    return {
        "final": pd.concat(final_parts, ignore_index=True).sort_values("date").reset_index(drop=True),
        "refit_log": pd.DataFrame(refit_rows),
        "init_table": pd.concat(init_parts, ignore_index=True) if init_parts else pd.DataFrame(),
        "state_table": pd.concat(state_parts, ignore_index=True) if state_parts else pd.DataFrame(),
        "freq": freq,
        "freq_label": FREQ_LABELS[freq],
        "initial_train_end": initial_train_end,
        "factor_start": pd.Timestamp(factors["date"].min()),
        "factor_end": pd.Timestamp(factors["date"].max()),
    }


def write_outputs(args: argparse.Namespace, result: dict[str, object]) -> Path:
    subdir = args.output_dir / f"{args.n_states}state"
    subdir.mkdir(parents=True, exist_ok=True)

    label = result["freq_label"]
    stem = f"{args.n_states}state_feature_tuned_expanding_{label}"
    final = result["final"]
    refit_log = result["refit_log"]
    init_table = result["init_table"]
    state_table = result["state_table"]

    probability_path = subdir / f"market_state_probabilities_{stem}.csv"
    final.to_csv(probability_path, index=False, encoding="utf-8-sig")
    refit_log.to_csv(subdir / f"refit_log_{stem}.csv", index=False, encoding="utf-8-sig")
    if not init_table.empty:
        init_table.to_csv(subdir / f"initialization_comparison_{stem}.csv", index=False, encoding="utf-8-sig")
    if not state_table.empty:
        state_table.to_csv(subdir / f"state_table_{stem}.csv", index=False, encoding="utf-8-sig")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_DIR)),
        "n_states": args.n_states,
        "feature_columns": list(TUNED_FEATURE_COLUMNS),
        "refit_freq": result["freq"],
        "refit_label": label,
        "initial_train_end": str(pd.Timestamp(result["initial_train_end"]).date()),
        "min_train_obs": args.min_train_obs,
        "hmm_max_iter": args.hmm_max_iter,
        "factor_start": str(pd.Timestamp(result["factor_start"]).date()),
        "factor_end": str(pd.Timestamp(result["factor_end"]).date()),
        "output_probability_path": str(probability_path),
        "uses_expanding_training_window": True,
        "train_cutoff_rule": "date < segment_start",
        "uses_forward_filter_only": True,
        "uses_smoothed_probabilities": False,
    }
    (subdir / f"model_meta_{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return probability_path


def main() -> None:
    args = parse_args()
    result = run_expanding(args)
    probability_path = write_outputs(args, result)
    print(probability_path)


if __name__ == "__main__":
    main()
