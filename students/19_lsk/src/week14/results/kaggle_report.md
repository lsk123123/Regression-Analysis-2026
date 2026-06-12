# Real Data Report: Housing (Task D)

## 数据来源与背景
- **数据集:** Housing (housing.csv)
- **样本数:** 20640
- **特征数:** 12

## 诊断信息

- rank(X_train) = 12
- Condition Number = 2.37e+05
- 前 5 个 PC 解释方差比例 = 81.6%

## 测试集表现

| 方法 | RMSE | MAE | 复杂度 |
|------|------|-----|--------|
| OLS   | 68779.9116 | 50100.1462 | 12 |
| Lasso | 68779.9117 | 50100.1463 | 非零=12/12 |
| PCR   | 68779.9116 | 50100.1462 | k=12 |

## Lasso 系数 (Top-15 by |coef|)

| 特征 | 系数 |
|------|------|
| ocean_proximity_ISLAND | 135888.184026 |
| ocean_proximity_INLAND | -41076.058686 |
| median_income | 39269.682386 |
| longitude | -26502.202401 |
| latitude | -24915.560671 |
| ocean_proximity_NEAR BAY | -6066.045268 |
| ocean_proximity_NEAR OCEAN | 3173.012147 |
| housing_median_age | 1105.974808 |
| total_bedrooms | 105.721354 |
| households | 42.313209 |
| population | -37.326969 |
| total_rooms | -5.872325 |

## 真实数据解释

### OLS 是否出现了高维/共线性不稳定迹象？

Condition Number = 237291.7，存在中等及以上共线性。

### Lasso 与 PCR 谁表现更好？为什么？

PCR (RMSE=68779.9116) 优于 Lasso (RMSE=68779.9117)。
数据可能更接近 latent-factor truth。

### 这份数据到底适合筛选还是压缩？

前 5 个 PC 仅解释 81.6% 方差 → 信息分散，偏 **筛选 (Lasso)**。
最终选择取决于业务需求：可解释的变量名单 → Lasso；稳定预测 → PCR。
