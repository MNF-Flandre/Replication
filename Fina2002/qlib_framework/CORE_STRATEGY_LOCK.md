# Gap x Market State Core Strategy Lock

## Locked Core

当前锁定的核心版本是：

```text
qlib_framework/run_core_cs500_gap_state_strategy.py
```

## Locked HMM

The main line now uses one HMM recognizer only:

```text
market_state_model/output/feature_tuned_expanding_hmm_2015/4state/market_state_probabilities_4state_feature_tuned_expanding_quarterly.csv
```

This is the 2015-ready expanding 4-state HMM. `run_core_cs500_gap_state_strategy.py`
uses it as `DEFAULT_HMM_PATH`; the older direct full-sample 4-state file is kept
only as `LEGACY_DIRECT_HMM_PATH` for archived comparison work.

核心输出目录是：

```text
qlib_framework/output/core_cs500_gap_state_strategy/
```

最新长周期输出目录是：

```text
qlib_framework/output/core_cs500_locked_expanding_hmm_gap_D_2015_2024/
```

核心策略口径：

- 指数：中证500 `000905` buy-and-hold。
- 高Gap：在中证500成分内按 `leverage_gap` 选择最高 20%。
- 市场状态择时：市场好时持有中证500，市场差时空仓。
- 高Gap+择时：市场好时持有高Gap组合，市场差时空仓。

市场状态规则：

- 使用四状态 HMM forward-filtered state。
- `L+` 和 `Stable` 视为 risk-on。
- `H` 和 `L-` 视为 risk-off。
- 用上一交易日状态决定当日持仓。
- 最小持有期为 5 个交易日。

数据边界：

- 不重新计算 `d*`。
- 不重新估计 `phi0`。
- 不重新训练 HMM。
- 不使用 smoothed probabilities。
- 不使用行业轮动或板块选择。
- 高Gap 严格指 `observed_debt_ratio - optimal_debt_ratio`。

## Current Result Snapshot

使用唯一锁定的 expanding 4-state HMM，并在 2015-01-01 至 2024-12-31 长周期窗口重新运行核心脚本，核心结果可复现。

| 线 | 净累计收益 | 相对指数 | 最大回撤 | Active IR |
|---|---:|---:|---:|---:|
| 指数 | 26.75% | 0.00% | -66.15% | NA |
| 高Gap | 23.92% | -2.83% | -66.86% | -0.01 |
| 纯HMM | 75.07% | 48.32% | -35.54% | 0.04 |
| HMM+高Gap | 136.36% | 109.61% | -35.02% | 0.17 |

锁定核心策略为 `gap_state_core`，即“市场好时买高Gap，市场差时空仓”。

## Cleanup

已删除早期探索脚本和验证结果目录，删除清单保存在：

```text
qlib_framework/output/core_cs500_gap_state_strategy/cleanup_deleted_validation_artifacts.csv
```

当前保留的主要脚本：

- `qlib_framework/run_core_cs500_gap_state_strategy.py`
- `qlib_framework/control_hs300_gap_state_turnover.py`
- `qlib_framework/build_hs300_gap_state_recognizer.py`
- `qlib_framework/prepare_qlib_data.py`
- `qlib_framework/smoke_test.py`

## Next Robustness Tests

后续稳健性检验围绕核心版本展开，不再把探索脚本作为主线。

1. 交易费率：0、0.03%、0.05%、0.10%。
2. 高Gap分位：top 10%、20%、30%。
3. Gap定义：横截面排序、`d_obs - d*` 固定阈值、`|d_obs - d*|` 诊断口径。
4. PIT新鲜度：365天、540天、full。
5. 最小持有期：3、5、10、20个交易日。
6. Risk-off腿：空仓 vs 无高Gap组合。
7. 指数池：沪深300、中证500、中证1000、中证800、全A。
8. 锁定 HMM 口径：固定使用 expanding 4-state HMM；旧 HMM 只做归档对照，不作为主线稳健性。
9. 子区间：2021、2022、2023、2024，以及回撤/修复窗口。
10. Gap刷新频率：日频PIT、月频、季频。
