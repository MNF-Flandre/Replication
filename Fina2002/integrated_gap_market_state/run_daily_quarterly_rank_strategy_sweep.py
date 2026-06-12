from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DAILY_PANEL_PATH = OUTPUT_DIR / "daily_pit_gap_market_state_panel.parquet"
ALPHAAGENT_DAILY_PATH = Path(
    os.environ.get("QUANT_ALPHAAGENT_DAILY_CSV", PROJECT_ROOT / "external_data" / "alphaagent_qlib_daily_ashare.csv")
).expanduser()

OUT_SWEEP = OUTPUT_DIR / "daily_quarterly_rank_gap_strategy_sweep.csv"
OUT_BEST = OUTPUT_DIR / "daily_quarterly_rank_gap_strategy_best.csv"
OUT_REPORT = OUTPUT_DIR / "daily_quarterly_rank_gap_strategy_sweep_report.md"

PANEL_COLUMNS = [
    "firm_id",
    "trade_date",
    "days_since_available",
    "leverage_gap",
    "market_regime",
    "entry_score_mean20",
    "bullish_transition_past20",
    "bearish_transition_past20",
]

SAMPLES = {
    "full": None,
    "fresh_365": 365,
    "fresh_540": 540,
}
LIQUIDITY_POOLS = {
    "all": None,
    "amount20_top80": 0.80,
    "amount20_top60": 0.60,
    "amount20_top40": 0.40,
}
RANK_RULES = {
    "pct_10": ("pct", 0.10),
    "pct_20": ("pct", 0.20),
    "pct_30": ("pct", 0.30),
    "fixed_50": ("fixed", 50),
    "fixed_100": ("fixed", 100),
    "fixed_200": ("fixed", 200),
}
BEHAVIORS = [
    "priority_expansion_first",
    "mutual_exclusive",
    "recovery_only_hml",
    "expansion_only_lmh",
    "always_hml",
    "always_lmh",
]

COST_BPS = 10.0
TRADING_DAYS = 252


@dataclass(frozen=True)
class SelectedGroups:
    high: tuple[int, ...]
    low: tuple[int, ...]
    universe_size: int
    amount20_median: float
    gap_high_mean: float
    gap_low_mean: float


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [col for col in required if col not in columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_daily_panel() -> pd.DataFrame:
    panel = pd.read_parquet(DAILY_PANEL_PATH, columns=PANEL_COLUMNS)
    require_columns(panel.columns.tolist(), PANEL_COLUMNS, "daily PIT panel")
    panel["firm_id"] = panel["firm_id"].astype("int32")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["leverage_gap"] = pd.to_numeric(panel["leverage_gap"], errors="coerce")
    panel["days_since_available"] = pd.to_numeric(
        panel["days_since_available"], errors="coerce"
    )
    return panel


def load_stock_returns(panel: pd.DataFrame) -> pd.DataFrame:
    header = pd.read_csv(ALPHAAGENT_DAILY_PATH, nrows=0).columns.tolist()
    require_columns(
        header,
        ["date", "code", "return", "amount", "turn"],
        "alphaagent qlib daily stock file",
    )
    stock = pd.read_csv(
        ALPHAAGENT_DAILY_PATH,
        usecols=["date", "code", "return", "amount", "turn"],
    )
    stock["firm_id"] = stock["code"].astype(str).str.extract(r"(\d{6})$", expand=False)
    stock["firm_id"] = pd.to_numeric(stock["firm_id"], errors="coerce")
    stock["trade_date"] = pd.to_datetime(stock["date"], errors="coerce")
    stock["stock_return"] = pd.to_numeric(stock["return"], errors="coerce")
    stock["amount"] = pd.to_numeric(stock["amount"], errors="coerce")
    stock["turn"] = pd.to_numeric(stock["turn"], errors="coerce")
    stock = stock.dropna(subset=["firm_id", "trade_date"])
    stock["firm_id"] = stock["firm_id"].astype("int32")

    min_date = panel["trade_date"].min()
    max_date = panel["trade_date"].max()
    firm_ids = set(panel["firm_id"].unique())
    stock = stock.loc[
        stock["firm_id"].isin(firm_ids)
        & (stock["trade_date"] >= min_date)
        & (stock["trade_date"] <= max_date)
    ].copy()
    stock = stock.sort_values(["firm_id", "trade_date"]).drop_duplicates(
        ["firm_id", "trade_date"], keep="last"
    )
    stock["next_ret_1d"] = stock.groupby("firm_id", sort=False)["stock_return"].shift(-1)
    stock["amount20"] = (
        stock.groupby("firm_id", sort=False)["amount"]
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return stock[["firm_id", "trade_date", "next_ret_1d", "amount20", "turn"]]


def build_state_table(panel: pd.DataFrame) -> pd.DataFrame:
    state = (
        panel[
            [
                "trade_date",
                "market_regime",
                "entry_score_mean20",
                "bullish_transition_past20",
                "bearish_transition_past20",
            ]
        ]
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    entry_hist = state["entry_score_mean20"].shift(1)
    state["entry_top80_trailing252"] = entry_hist.rolling(
        252, min_periods=60
    ).quantile(0.80)
    state["entry_bottom20_trailing252"] = entry_hist.rolling(
        252, min_periods=60
    ).quantile(0.20)
    state["risk_recovery"] = (
        (
            state["entry_score_mean20"]
            >= state["entry_top80_trailing252"]
        )
        | state["bullish_transition_past20"].astype(bool)
        | state["market_regime"].eq("L+")
    )
    state["risk_expansion"] = (
        (
            state["entry_score_mean20"]
            <= state["entry_bottom20_trailing252"]
        )
        | state["bearish_transition_past20"].astype(bool)
        | state["market_regime"].isin(["H", "L-"])
    )
    return state


def quarterly_rank_dates(state: pd.DataFrame) -> list[pd.Timestamp]:
    dates = state[["trade_date"]].copy()
    dates["quarter"] = dates["trade_date"].dt.to_period("Q")
    rank_dates = dates.groupby("quarter", sort=True)["trade_date"].max().tolist()
    return [pd.Timestamp(d) for d in rank_dates]


def select_groups(
    cross_section: pd.DataFrame,
    sample: str,
    liquidity_pool: str,
    rank_rule: str,
) -> SelectedGroups | None:
    cs = cross_section.dropna(subset=["firm_id", "leverage_gap"]).copy()
    max_days = SAMPLES[sample]
    if max_days is not None:
        cs = cs.loc[cs["days_since_available"] <= max_days]
    top_liq = LIQUIDITY_POOLS[liquidity_pool]
    if top_liq is not None:
        cs = cs.dropna(subset=["amount20"])
        if cs.empty:
            return None
        threshold = cs["amount20"].quantile(1.0 - top_liq)
        cs = cs.loc[cs["amount20"] >= threshold]
    cs = cs.sort_values(["leverage_gap", "firm_id"]).reset_index(drop=True)
    n = len(cs)
    if n < 60:
        return None
    kind, value = RANK_RULES[rank_rule]
    if kind == "pct":
        k = int(np.floor(n * float(value)))
    else:
        k = int(min(int(value), np.floor(n / 3)))
    k = max(k, 10)
    if 2 * k > n:
        k = int(np.floor(n / 2))
    if k <= 0:
        return None
    low = cs.head(k)
    high = cs.tail(k)
    return SelectedGroups(
        high=tuple(high["firm_id"].astype(int).tolist()),
        low=tuple(low["firm_id"].astype(int).tolist()),
        universe_size=int(n),
        amount20_median=float(cs["amount20"].median()) if "amount20" in cs else np.nan,
        gap_high_mean=float(high["leverage_gap"].mean()),
        gap_low_mean=float(low["leverage_gap"].mean()),
    )


def build_selection_cache(
    rank_panel: pd.DataFrame,
    rank_dates: list[pd.Timestamp],
) -> dict[tuple[pd.Timestamp, str, str, str], SelectedGroups]:
    by_date = {
        pd.Timestamp(date): part
        for date, part in rank_panel.groupby("trade_date", sort=True)
    }
    cache: dict[tuple[pd.Timestamp, str, str, str], SelectedGroups] = {}
    for rank_date in rank_dates:
        cross = by_date.get(rank_date)
        if cross is None:
            continue
        for sample in SAMPLES:
            for liquidity_pool in LIQUIDITY_POOLS:
                for rank_rule in RANK_RULES:
                    selected = select_groups(cross, sample, liquidity_pool, rank_rule)
                    if selected is not None:
                        cache[(rank_date, sample, liquidity_pool, rank_rule)] = selected
    return cache


def direction_for_behavior(recovery: bool, expansion: bool, behavior: str) -> str:
    if behavior == "priority_expansion_first":
        if expansion:
            return "LowMinusHigh"
        if recovery:
            return "HighMinusLow"
        return "cash"
    if behavior == "mutual_exclusive":
        if recovery and not expansion:
            return "HighMinusLow"
        if expansion and not recovery:
            return "LowMinusHigh"
        return "cash"
    if behavior == "recovery_only_hml":
        return "HighMinusLow" if recovery else "cash"
    if behavior == "expansion_only_lmh":
        return "LowMinusHigh" if expansion else "cash"
    if behavior == "always_hml":
        return "HighMinusLow"
    if behavior == "always_lmh":
        return "LowMinusHigh"
    raise ValueError(f"Unknown behavior: {behavior}")


def weights_for_direction(selected: SelectedGroups | None, direction: str) -> dict[int, float]:
    if selected is None or direction == "cash":
        return {}
    high = selected.high
    low = selected.low
    if not high or not low:
        return {}
    if direction == "HighMinusLow":
        long_names, short_names = high, low
    elif direction == "LowMinusHigh":
        long_names, short_names = low, high
    else:
        raise ValueError(f"Unknown direction: {direction}")
    weights: dict[int, float] = {}
    for firm_id in long_names:
        weights[int(firm_id)] = weights.get(int(firm_id), 0.0) + 1.0 / len(long_names)
    for firm_id in short_names:
        weights[int(firm_id)] = weights.get(int(firm_id), 0.0) - 1.0 / len(short_names)
    return weights


def turnover_between(prev: dict[int, float], current: dict[int, float]) -> float:
    names = set(prev) | set(current)
    if not names:
        return 0.0
    return 0.5 * sum(abs(current.get(name, 0.0) - prev.get(name, 0.0)) for name in names)


def portfolio_return(
    ret_matrix: pd.DataFrame,
    trade_date: pd.Timestamp,
    selected: SelectedGroups | None,
    direction: str,
) -> tuple[float, int, int]:
    if selected is None or direction == "cash":
        return 0.0, 0, 0
    if trade_date not in ret_matrix.index:
        return np.nan, 0, 0
    row = ret_matrix.loc[trade_date]
    high_ret = row.reindex(selected.high).dropna()
    low_ret = row.reindex(selected.low).dropna()
    if high_ret.empty or low_ret.empty:
        return np.nan, int(high_ret.count()), int(low_ret.count())
    hml = float(high_ret.mean() - low_ret.mean())
    if direction == "HighMinusLow":
        return hml, int(high_ret.count()), int(low_ret.count())
    if direction == "LowMinusHigh":
        return -hml, int(high_ret.count()), int(low_ret.count())
    raise ValueError(f"Unknown direction: {direction}")


def max_drawdown(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    wealth = np.cumprod(1.0 + r.to_numpy(dtype="float64"))
    wealth = np.concatenate([[1.0], wealth])
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def summarize_returns(series: pd.Series) -> dict[str, float | int]:
    r = pd.to_numeric(series, errors="coerce").dropna()
    if r.empty:
        return {
            "n_days": 0,
            "mean_daily": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "win_rate": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
        }
    cum = float(np.prod(1.0 + r) - 1.0)
    ann_return = float((1.0 + cum) ** (TRADING_DAYS / len(r)) - 1.0) if cum > -1 else np.nan
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "n_days": int(r.count()),
        "mean_daily": float(r.mean()),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float((r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS))
        if r.std(ddof=1) > 0
        else np.nan,
        "win_rate": float((r > 0).mean()),
        "cumulative_return": cum,
        "max_drawdown": max_drawdown(r),
    }


def simulate_combo(
    state: pd.DataFrame,
    ret_matrix: pd.DataFrame,
    rank_date_by_day: dict[pd.Timestamp, pd.Timestamp],
    selection_cache: dict[tuple[pd.Timestamp, str, str, str], SelectedGroups],
    sample: str,
    liquidity_pool: str,
    rank_rule: str,
    behavior: str,
) -> dict[str, float | int | str]:
    gross_returns = []
    net_returns = []
    turnovers = []
    active_days = 0
    recovery_days = 0
    expansion_days = 0
    overlap_days = 0
    hml_days = 0
    lmh_days = 0
    cash_days = 0
    missing_return_days = 0
    high_counts = []
    low_counts = []
    universe_sizes = []
    gap_high_means = []
    gap_low_means = []
    prev_weights: dict[int, float] = {}
    weight_cache: dict[tuple[pd.Timestamp, str], dict[int, float]] = {}

    for row in state.itertuples(index=False):
        trade_date = pd.Timestamp(row.trade_date)
        recovery = bool(row.risk_recovery)
        expansion = bool(row.risk_expansion)
        direction = direction_for_behavior(recovery, expansion, behavior)
        rank_date = rank_date_by_day.get(trade_date)
        selected = (
            selection_cache.get((rank_date, sample, liquidity_pool, rank_rule))
            if rank_date is not None
            else None
        )
        if selected is None:
            direction = "cash"
        weights_key = (rank_date, direction)
        if direction == "cash":
            current_weights = {}
        else:
            current_weights = weight_cache.get(weights_key)
            if current_weights is None:
                current_weights = weights_for_direction(selected, direction)
                weight_cache[weights_key] = current_weights
        turnover = turnover_between(prev_weights, current_weights)
        prev_weights = current_weights

        gross, high_count, low_count = portfolio_return(
            ret_matrix, trade_date, selected, direction
        )
        if np.isnan(gross):
            missing_return_days += 1
            gross = 0.0
        cost = (COST_BPS / 10000.0) * turnover
        net = gross - cost
        gross_returns.append(gross)
        net_returns.append(net)
        turnovers.append(turnover)

        if recovery:
            recovery_days += 1
        if expansion:
            expansion_days += 1
        if recovery and expansion:
            overlap_days += 1
        if direction == "HighMinusLow":
            hml_days += 1
            active_days += 1
        elif direction == "LowMinusHigh":
            lmh_days += 1
            active_days += 1
        else:
            cash_days += 1
        if selected is not None:
            high_counts.append(high_count)
            low_counts.append(low_count)
            universe_sizes.append(selected.universe_size)
            gap_high_means.append(selected.gap_high_mean)
            gap_low_means.append(selected.gap_low_mean)

    gross_series = pd.Series(gross_returns, dtype="float64")
    net_series = pd.Series(net_returns, dtype="float64")
    gross_stats = summarize_returns(gross_series)
    net_stats = summarize_returns(net_series)
    turnover_series = pd.Series(turnovers, dtype="float64")
    return {
        "sample": sample,
        "liquidity_pool": liquidity_pool,
        "rank_rule": rank_rule,
        "behavior": behavior,
        "cost_bps": COST_BPS,
        "n_days": int(len(state)),
        "active_days": active_days,
        "active_ratio": active_days / len(state) if len(state) else np.nan,
        "hml_days": hml_days,
        "lmh_days": lmh_days,
        "cash_days": cash_days,
        "recovery_days": recovery_days,
        "expansion_days": expansion_days,
        "overlap_days": overlap_days,
        "missing_return_days": missing_return_days,
        "avg_turnover": float(turnover_series.mean()) if len(turnover_series) else np.nan,
        "median_turnover": float(turnover_series.median()) if len(turnover_series) else np.nan,
        "total_turnover": float(turnover_series.sum()) if len(turnover_series) else np.nan,
        "avg_high_return_members": float(np.nanmean(high_counts)) if high_counts else np.nan,
        "avg_low_return_members": float(np.nanmean(low_counts)) if low_counts else np.nan,
        "avg_universe_size": float(np.nanmean(universe_sizes)) if universe_sizes else np.nan,
        "avg_gap_high_mean": float(np.nanmean(gap_high_means)) if gap_high_means else np.nan,
        "avg_gap_low_mean": float(np.nanmean(gap_low_means)) if gap_low_means else np.nan,
        **{f"gross_{key}": value for key, value in gross_stats.items()},
        **{f"net_{key}": value for key, value in net_stats.items()},
    }


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无可用数据_"
    view = df if max_rows is None else df.head(max_rows)
    return view.to_markdown(index=False, floatfmt=".6f")


def write_report(results: pd.DataFrame, state: pd.DataFrame) -> None:
    best_net = results.sort_values(
        ["net_sharpe", "net_ann_return", "net_max_drawdown"],
        ascending=[False, False, False],
    ).head(20)
    best_state_dependent = (
        results.loc[
            results["behavior"].isin(["priority_expansion_first", "mutual_exclusive"])
        ]
        .sort_values(
            ["net_sharpe", "net_ann_return", "net_max_drawdown"],
            ascending=[False, False, False],
        )
        .head(20)
    )
    best_no_always = (
        results.loc[~results["behavior"].isin(["always_hml", "always_lmh"])]
        .sort_values(
            ["net_sharpe", "net_ann_return", "net_max_drawdown"],
            ascending=[False, False, False],
        )
        .head(20)
    )
    best_by_behavior = (
        results.sort_values("net_sharpe", ascending=False)
        .groupby("behavior", sort=False)
        .head(5)
        .sort_values(["behavior", "net_sharpe"], ascending=[True, False])
    )
    best_by_sample = (
        results.sort_values("net_sharpe", ascending=False)
        .groupby("sample", sort=False)
        .head(5)
        .sort_values(["sample", "net_sharpe"], ascending=[True, False])
    )
    cols = [
        "sample",
        "liquidity_pool",
        "rank_rule",
        "behavior",
        "net_sharpe",
        "net_ann_return",
        "net_ann_vol",
        "net_max_drawdown",
        "net_win_rate",
        "net_cumulative_return",
        "avg_turnover",
        "active_ratio",
        "hml_days",
        "lmh_days",
        "cash_days",
        "avg_universe_size",
        "avg_high_return_members",
        "avg_low_return_members",
    ]
    state_counts = {
        "days": int(len(state)),
        "recovery_days": int(state["risk_recovery"].sum()),
        "expansion_days": int(state["risk_expansion"].sum()),
        "overlap_days": int((state["risk_recovery"] & state["risk_expansion"]).sum()),
        "neutral_days": int((~(state["risk_recovery"] | state["risk_expansion"])).sum()),
    }
    overall_best = best_net[cols].head(1)
    state_best = best_state_dependent[cols].head(1)
    no_always_best = best_no_always[cols].head(1)
    lines = [
        "# 日频状态切换 + 季度 Gap 排名策略扫描报告",
        "",
        "## 实验设置",
        "",
        "本实验按季度使用 PIT `leverage_gap = observed_debt_ratio - optimal_debt_ratio` 对股票池内企业排序；季度内不重新按 Gap 排名。每天只读取当日 HMM 市场状态，决定持有 `HighMinusLow`、`LowMinusHigh` 或现金。收益使用下一交易日个股收益，避免使用当天状态赚当天收益。",
        "",
        "没有行业筛选、行业打分或板块轮动。HighGap 严格是高 Gap，不是普通 observed leverage。",
        "",
        "参数扫描范围：",
        "",
        "- stale 股票池：`full`、`fresh_365`、`fresh_540`。",
        "- 流动性股票池：`all`、按 rank date 过去 20 日成交额均值保留 top 80%、top 60%、top 40%。",
        "- Gap 分组：两端各 10%、20%、30%，以及固定两端各 50、100、200 只。",
        "- 交易行为：扩张优先双向切换、互斥双向切换、只做 recovery HML、只做 expansion LMH、always HML、always LMH benchmark。",
        f"- 成本：报告同时计算 gross 和扣除 {COST_BPS:.0f} bps 单位换手成本后的 net，排序以 `net_sharpe` 为主。",
        "",
        "市场状态日频触发统计：",
        "",
        markdown_table(pd.DataFrame([state_counts])),
        "",
        "## 结论摘要",
        "",
        "按扣除 10 bps 单位换手成本后的 `net_sharpe` 排序，当前样本内全网格最优组合如下：",
        "",
        markdown_table(overall_best),
        "",
        "如果只看真正使用 daily market state 切换方向的双向策略，即 `priority_expansion_first` 和 `mutual_exclusive`，最优组合如下：",
        "",
        markdown_table(state_best),
        "",
        "如果排除 `always_hml` / `always_lmh` 两个纯 benchmark，但允许只做 recovery 或只做 expansion，最优组合如下：",
        "",
        markdown_table(no_always_best),
        "",
        "## 全部组合中按 net Sharpe 排名前 20",
        "",
        markdown_table(best_net[cols], max_rows=20),
        "",
        "## 只看双向状态切换策略的前 20",
        "",
        markdown_table(best_state_dependent[cols], max_rows=20),
        "",
        "## 排除 always benchmark 后的前 20",
        "",
        markdown_table(best_no_always[cols], max_rows=20),
        "",
        "## 各交易行为下的较优组合",
        "",
        markdown_table(best_by_behavior[cols], max_rows=40),
        "",
        "## 各 stale 样本下的较优组合",
        "",
        markdown_table(best_by_sample[cols], max_rows=30),
        "",
        "## 解释和限制",
        "",
        "- 这是样本内参数扫描，用来定位策略形态，不应直接作为论文主结论或实盘参数。",
        "- 如果 always benchmark 排名靠前，说明当前状态切换规则未必增加择时价值，需要在样本外或滚动验证中继续检验。",
        "- `risk_expansion` 因包含 `market_regime in {H, L-}`，覆盖天数通常较高；扩张优先规则会使组合大部分时间偏向 `LowMinusHigh`。",
        "- 日频状态切换会带来更高换手；报告中的 net 结果已经扣除简化换手成本，但尚未处理卖空约束、停牌成交约束和冲击成本。",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading PIT daily panel...")
    panel = load_daily_panel()
    print("Loading stock returns and liquidity...")
    stock = load_stock_returns(panel)
    print("Merging returns/liquidity into panel...")
    data = panel.merge(stock, on=["firm_id", "trade_date"], how="left")
    state = build_state_table(panel)
    rank_dates = quarterly_rank_dates(state)
    rank_panel = data.loc[data["trade_date"].isin(rank_dates)].copy()
    print("Building quarterly selection cache...")
    selection_cache = build_selection_cache(rank_panel, rank_dates)

    ret_matrix = (
        stock.pivot(index="trade_date", columns="firm_id", values="next_ret_1d")
        .sort_index()
        .astype("float32")
    )
    rank_map = pd.DataFrame({"trade_date": state["trade_date"]}).sort_values("trade_date")
    rank_df = pd.DataFrame({"rank_date": rank_dates}).sort_values("rank_date")
    rank_map = pd.merge_asof(
        rank_map,
        rank_df,
        left_on="trade_date",
        right_on="rank_date",
        direction="backward",
    )
    rank_date_by_day = {
        pd.Timestamp(row.trade_date): pd.Timestamp(row.rank_date)
        for row in rank_map.dropna(subset=["rank_date"]).itertuples(index=False)
    }

    rows = []
    total = len(SAMPLES) * len(LIQUIDITY_POOLS) * len(RANK_RULES) * len(BEHAVIORS)
    done = 0
    for sample in SAMPLES:
        for liquidity_pool in LIQUIDITY_POOLS:
            for rank_rule in RANK_RULES:
                for behavior in BEHAVIORS:
                    done += 1
                    if done % 25 == 0 or done == 1:
                        print(f"Simulating {done}/{total}...")
                    rows.append(
                        simulate_combo(
                            state,
                            ret_matrix,
                            rank_date_by_day,
                            selection_cache,
                            sample,
                            liquidity_pool,
                            rank_rule,
                            behavior,
                        )
                    )
    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["net_sharpe", "net_ann_return", "net_max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    results.to_csv(OUT_SWEEP, index=False, encoding="utf-8-sig")
    results.head(50).to_csv(OUT_BEST, index=False, encoding="utf-8-sig")
    write_report(results, state)
    print("Done.")
    print("Sweep results:", results.shape)
    print("Best net Sharpe:")
    print(
        results[
            [
                "sample",
                "liquidity_pool",
                "rank_rule",
                "behavior",
                "net_sharpe",
                "net_ann_return",
                "net_max_drawdown",
                "avg_turnover",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
