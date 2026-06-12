# 最优负债率模型实现文档

## 1. 实现目标

本模块用于在企业-时间维度上计算理论最优负债率，并进一步构造资本结构偏离变量。

核心输出为：

$$
d^*_{i,t}
=
\left[
\frac{
\tau_{i,t}\left((1+r_{i,t})^T-1\right)
}{
\phi_{i,t}T\eta
}
\right]^{\frac{1}{\eta-1}}
$$

以及：

$$
Gap_{i,t}=d^{obs}_{i,t}-d^*_{i,t}.
$$

其中：

- $d^*_{i,t}$：理论最优负债率；
- $d^{obs}_{i,t}$：企业实际负债率；
- $Gap_{i,t}$：资本结构偏离程度。

该模块不直接预测收益率，也不直接解释企业估值。它提供的是一个可计算、可比较的资本结构基准。

## 2. 模型输入

最小输入表以企业 $i$、时间 $t$ 为索引，建议包含以下字段：

| 字段 | 记号 | 类型 | 含义 |
| --- | --- | --- | --- |
| `firm_id` | $i$ | string/int | 企业标识 |
| `date` | $t$ | date | 观测日期或财报期 |
| `total_assets` | $K_{i,t}$ | float | 企业资产规模 |
| `total_debt` | $D_{i,t}$ | float | 企业债务规模 |
| `tax_rate` | $\tau_{i,t}$ | float | 有效税率 |
| `debt_cost` | $r_{i,t}$ | float | 债务利率或融资成本 |
| `distress_intensity` | $\phi_{i,t}$ | float | 有效财务困境强度 |

全局配置参数：

| 参数 | 记号 | 默认建议 | 含义 |
| --- | --- | --- | --- |
| `horizon` | $T$ | 1 或 3 | 评估期限 |
| `eta` | $\eta$ | 大于 1 | 财务困境成本相对负债率的曲率 |
| `debt_ratio_upper` | $\bar d$ | 1.0 | 负债率上界 |

## 3. 变量定义

实际负债率定义为：

$$
d^{obs}_{i,t}=\frac{D_{i,t}}{K_{i,t}}.
$$

其中 $K_{i,t}>0$。若资产规模缺失、为零或为负，应将该样本标记为不可计算。

有效财务困境强度 $\phi_{i,t}$ 是一个综合风险强度参数。它吸收企业财务困境发生可能性和经济严重程度，要求：

$$
\phi_{i,t}>0.
$$

实现时可以直接使用外部给定的 $\phi_{i,t}$，也可以由代理变量估计：

$$
\phi_{i,t}=\exp(X_{i,t}'\beta).
$$

指数形式的好处是保证 $\phi_{i,t}$ 始终为正。

## 4. 核心计算流程

### 4.1 计算未裁剪的理论负债率

先计算：

$$
\tilde d_{i,t}
=
\left[
\frac{
\tau_{i,t}\left((1+r_{i,t})^T-1\right)
}{
\phi_{i,t}T\eta
}
\right]^{\frac{1}{\eta-1}}.
$$

实现中需要检查以下条件：

- $\tau_{i,t}\ge 0$；
- $r_{i,t}>-1$；
- $\phi_{i,t}>0$；
- $T>0$；
- $\eta>1$。

如果分子为 0，则 $\tilde d_{i,t}=0$。如果输入不满足约束，应返回缺失值或记录错误标记。

### 4.2 加入边界约束

实证实现中，负债率应限制在合理区间：

$$
d^*_{i,t}
=
\min\{\bar d,\max\{0,\tilde d_{i,t}\}\}.
$$

默认可取 $\bar d=1$。如果研究对象允许账面负债率超过 1，也可以将上界设为行业分位数或固定值，例如 1.5。

### 4.3 计算资本结构偏离

$$
Gap_{i,t}=d^{obs}_{i,t}-d^*_{i,t}.
$$

解释规则：

- $Gap_{i,t}>0$：企业实际负债率高于理论最优水平，属于 over-levered；
- $Gap_{i,t}<0$：企业实际负债率低于理论最优水平，属于 under-levered；
- $Gap_{i,t}\approx 0$：企业资本结构接近模型基准。

## 5. 推荐函数接口

```python
def compute_optimal_leverage(
    data,
    horizon: float,
    eta: float,
    debt_ratio_upper: float = 1.0,
):
    """
    输入企业-时间面板数据，输出理论最优负债率和资本结构偏离。

    Required columns:
        firm_id, date, total_assets, total_debt,
        tax_rate, debt_cost, distress_intensity

    Returns:
        原数据 + observed_debt_ratio + optimal_debt_ratio_raw
        + optimal_debt_ratio + leverage_gap + leverage_status
    """
```

推荐输出字段：

| 字段 | 含义 |
| --- | --- |
| `observed_debt_ratio` | $d^{obs}_{i,t}$ |
| `optimal_debt_ratio_raw` | 未裁剪的 $\tilde d_{i,t}$ |
| `optimal_debt_ratio` | 裁剪后的 $d^*_{i,t}$ |
| `leverage_gap` | $Gap_{i,t}$ |
| `leverage_status` | `over_levered` / `under_levered` / `near_optimal` |

## 6. 参数估计与代理变量

### 6.1 有效税率

有效税率可用：

$$
\tau_{i,t}=\frac{\text{income tax expense}_{i,t}}{\text{pretax income}_{i,t}}.
$$

实证中应进行合理裁剪，例如限制在 $[0,\tau^{statutory}]$ 或 $[0,1]$。若税前利润为负，可将有效税率设为 0 或缺失，具体取决于研究设计。

### 6.2 债务利率

债务利率可用：

$$
r_{i,t}=\frac{\text{interest expense}_{i,t}}{\text{interest-bearing debt}_{i,t-1}}.
$$

若无法获得企业层面的债务成本，可使用行业平均融资成本、信用利差加无风险利率，或企业债收益率代理。

### 6.3 有效财务困境强度

$\phi_{i,t}$ 可以由以下变量估计或代理：

| 维度 | 变量例子 |
| --- | --- |
| 盈利能力 | ROA、经营现金流、亏损虚拟变量 |
| 偿债压力 | 利息保障倍数、短债占比、融资成本 |
| 波动性 | 股票收益波动率、现金流波动率 |
| 资产结构 | 有形资产比例、抵押能力 |
| 外部环境 | 行业景气度、信用利差、宏观压力 |

一个可实现的参数化方式为：

$$
\phi_{i,t}=\exp(\beta_0+\beta'X_{i,t}).
$$

其中 $X_{i,t}$ 应统一方向，使数值越高代表财务困境压力越高。

## 7. 实现伪代码

```python
def compute_optimal_leverage(df, horizon, eta, debt_ratio_upper=1.0):
    assert horizon > 0
    assert eta > 1
    assert debt_ratio_upper > 0

    out = df.copy()

    out["observed_debt_ratio"] = out["total_debt"] / out["total_assets"]

    numerator = out["tax_rate"] * ((1.0 + out["debt_cost"]) ** horizon - 1.0)
    denominator = out["distress_intensity"] * horizon * eta

    valid = (
        (out["total_assets"] > 0)
        & (out["tax_rate"] >= 0)
        & (out["debt_cost"] > -1)
        & (out["distress_intensity"] > 0)
        & (denominator > 0)
        & (numerator >= 0)
    )

    out["optimal_debt_ratio_raw"] = np.nan
    out.loc[valid, "optimal_debt_ratio_raw"] = (
        numerator.loc[valid] / denominator.loc[valid]
    ) ** (1.0 / (eta - 1.0))

    out["optimal_debt_ratio"] = out["optimal_debt_ratio_raw"].clip(
        lower=0.0,
        upper=debt_ratio_upper,
    )

    out["leverage_gap"] = (
        out["observed_debt_ratio"] - out["optimal_debt_ratio"]
    )

    out["leverage_status"] = np.select(
        [
            out["leverage_gap"] > 0,
            out["leverage_gap"] < 0,
        ],
        [
            "over_levered",
            "under_levered",
        ],
        default="near_optimal",
    )

    return out
```

## 8. 校验项

实现完成后至少检查：

1. 所有可计算样本满足 $d^*_{i,t}\in[0,\bar d]$。
2. 当 $\tau_{i,t}$ 上升且其他变量不变时，$d^*_{i,t}$ 上升。
3. 当 $\phi_{i,t}$ 上升且其他变量不变时，$d^*_{i,t}$ 下降。
4. 当 $\eta \le 1$ 时，程序应拒绝计算。
5. 缺失值、负资产、异常融资成本不会生成无意义结果。

## 9. 下游使用方式

该模块的核心下游变量是：

$$
Gap_{i,t}=d^{obs}_{i,t}-d^*_{i,t}.
$$

可用于：

- 分组检验：over-levered 与 under-levered 企业的收益差异；
- 横截面回归：检验资本结构偏离对未来收益、风险或投资行为的解释力；
- 资产定价检验：将 $Gap_{i,t}$ 作为企业特征或构造因子；
- 动态调整分析：观察企业是否逐步向理论最优负债率回归。

