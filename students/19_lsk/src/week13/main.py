#!/usr/bin/env python3
"""
Week 13: Regularized Regression and Variable Selection
======================================================
Task A: 模拟共线性数据 → 正则化对比 → 变量筛选
Task B: housing.csv 真实数据 → 完整管道

单一入口: uv run src/week13/main.py
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # 无 GUI 后端
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- 复用自己的 utils ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.transformers import CustomStandardScaler
from src.utils.metrics import calculate_rmse, calculate_mae
from src.utils.models import AnalyticalOLS, forward_selection, backward_elimination

warnings.filterwarnings("ignore")

# ============================================================
# 路径常量
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_CSV = DATA_DIR / "synthetic_correlated.csv"
HOUSING_CSV   = DATA_DIR / "housing.csv"

SYNTHETIC_MD  = RESULTS_DIR / "synthetic_report.md"
KAGGLE_MD     = RESULTS_DIR / "kaggle_report.md"
SUMMARY_MD    = RESULTS_DIR / "summary_comparison.md"

# ============================================================
# 共享工具函数
# ============================================================
def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])

def make_correlated_regression_data(
    n_samples: int = 300,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple:
    """
    生成具有显式共线性的模拟回归数据。

    特征 (9 列, 截距不算):
      X0 : 均匀分布 [0, 10]           — 真实信号
      X1 : X0 * 2 + N(0, 0.5)        — 高度相关 (corr ~ 0.97)
      X2 : X0 * (-0.8) + N(0, 0.6)   — 高度相关 (corr ~ -0.9)
      X3 : N(0, 1)                    — 真实信号 (与 X0 无关)
      X4 : 纯噪声 N(0, 1)
      X5 : 纯噪声 N(0, 1)
      X6 : 纯噪声 N(0, 1)
      X7 : X3 * 1.5 + N(0, 0.3)      — 高度相关 (corr ~ 0.98)
      X8 : 纯噪声 N(0, 1)

    DGP: y = 3.0 + 2.0*X0 + 0.0*X1 + 0.0*X2 + 1.5*X3 + 0.0*(噪声) + ε
    """
    rng = np.random.default_rng(seed)
    X0 = rng.uniform(0, 10, n_samples)
    X1 = X0 * 2.0 + rng.normal(0, 0.5, n_samples)
    X2 = X0 * (-0.8) + rng.normal(0, 0.6, n_samples)
    X3 = rng.normal(0, 1, n_samples)
    X4 = rng.normal(0, 1, n_samples)
    X5 = rng.normal(0, 1, n_samples)
    X6 = rng.normal(0, 1, n_samples)
    X7 = X3 * 1.5 + rng.normal(0, 0.3, n_samples)
    X8 = rng.normal(0, 1, n_samples)

    X = np.column_stack([X0, X1, X2, X3, X4, X5, X6, X7, X8])
    eps = rng.normal(0, noise_std, n_samples)
    # DGP: 只依赖 X0 和 X3
    y = 3.0 + 2.0 * X0 + 1.5 * X3 + eps
    return X, y

# ============================================================
#                     Task A
# ============================================================
def task_a():
    print("\n" + "=" * 70)
    print("Task A: 模拟共线性数据 — 正则化与变量筛选")
    print("=" * 70)

    # ---- A1. 生成数据 ----
    X, y = make_correlated_regression_data(n_samples=500, noise_std=0.5, seed=42)
    feature_names = ["X0_true", "X1_corr+", "X2_corr-", "X3_true",
                     "X4_noise", "X5_noise", "X6_noise", "X7_corr+", "X8_noise"]

    # ---- A2. 保存 CSV ----
    df_syn = pd.DataFrame(X, columns=feature_names)
    df_syn["y"] = y
    df_syn.to_csv(SYNTHETIC_CSV, index=False)
    print(f"  模拟数据已保存: {SYNTHETIC_CSV}")

    # ---- A3.1 稳定性对比: OLS vs Ridge (50 次随机切分) ----
    print("\n--- A3.1 稳定性对比 (OLS vs Ridge, 50 splits) ---")
    n_splits = 50
    coefs_ols = []
    coefs_ridge = []
    scaler_stab = StandardScaler()   # Ridge 需要标准化

    for i in range(n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=i)

        # OLS
        ols = AnalyticalOLS()
        ols.fit(add_intercept(X_tr), y_tr)
        coefs_ols.append(ols.coef_[1:].copy())   # 去掉截距

        # Ridge (alpha = 10, 先标准化)
        X_tr_s = scaler_stab.fit_transform(X_tr)
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_s, y_tr)
        # 将系数变换回原始尺度：β_raw = β_scaled / σ_j
        beta_raw = ridge.coef_ / scaler_stab.scale_
        coefs_ridge.append(beta_raw)

    coefs_ols = np.array(coefs_ols)
    coefs_ridge = np.array(coefs_ridge)

    # 箱线图: 只展示高度相关族
    corr_family_idx = [0, 1, 2, 3, 7]   # X0, X1, X2, X3, X7
    corr_labels = [feature_names[i] for i in corr_family_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].boxplot(coefs_ols[:, corr_family_idx], labels=corr_labels, vert=True)
    axes[0].set_title("OLS Coefficients Across 50 Splits\n(Highly Correlated Features)")
    axes[0].axhline(y=0, color="gray", ls="--")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].boxplot(coefs_ridge[:, corr_family_idx], labels=corr_labels, vert=True)
    axes[1].set_title("Ridge Coefficients Across 50 Splits (alpha=10)\n(Highly Correlated Features)")
    axes[1].axhline(y=0, color="gray", ls="--")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stability_boxplot.png", dpi=150)
    plt.close(fig)
    print("  稳定性箱线图已保存: stability_boxplot.png")

    # 打印系数标准差对比
    ols_std = coefs_ols[:, corr_family_idx].std(axis=0)
    ridge_std = coefs_ridge[:, corr_family_idx].std(axis=0)
    print("\n  --- 系数标准差对比 (越低越稳定) ---")
    for name, os_, rs_ in zip(corr_labels, ols_std, ridge_std):
        print(f"  {name:<20s}: OLS std={os_:.4f}  Ridge std={rs_:.4f}")

    # ---- A3.2 Pipeline (使用自己的 Scaler) ----
    print("\n--- A3.2 Pipeline 搭建 ---")
    # 注：sklearn Pipeline 要求每一步有 fit/transform，我们的 CustomStandardScaler 符合此接口
    from sklearn.base import BaseEstimator, TransformerMixin

    class SklearnCompatibleScaler(BaseEstimator, TransformerMixin):
        """包装 CustomStandardScaler 使其兼容 sklearn Pipeline"""
        def __init__(self):
            self._scaler = CustomStandardScaler()
        def fit(self, X, y=None):
            self._scaler.fit(X)
            return self
        def transform(self, X, y=None):
            return self._scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---- A3.3 GridSearchCV 寻优 ----
    print("\n--- A3.3 GridSearchCV 超参数寻优 ---")
    alpha_grid = np.logspace(-4, 3, 50)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Ridge
    ridge_pipe = Pipeline([
        ("scaler", SklearnCompatibleScaler()),
        ("ridge", Ridge())
    ])
    ridge_gs = GridSearchCV(ridge_pipe, param_grid={"ridge__alpha": alpha_grid},
                            cv=kfold, scoring="neg_root_mean_squared_error")
    ridge_gs.fit(X_train, y_train)
    print(f"  Ridge 最优 alpha: {ridge_gs.best_params_['ridge__alpha']:.4f}")
    print(f"  Ridge 最优 CV RMSE: {-ridge_gs.best_score_:.4f}")

    # Lasso
    lasso_pipe = Pipeline([
        ("scaler", SklearnCompatibleScaler()),
        ("lasso", Lasso(max_iter=10000))
    ])
    lasso_gs = GridSearchCV(lasso_pipe, param_grid={"lasso__alpha": alpha_grid},
                            cv=kfold, scoring="neg_root_mean_squared_error")
    lasso_gs.fit(X_train, y_train)
    print(f"  Lasso 最优 alpha: {lasso_gs.best_params_['lasso__alpha']:.4f}")
    print(f"  Lasso 最优 CV RMSE: {-lasso_gs.best_score_:.4f}")

    # ElasticNet
    en_pipe = Pipeline([
        ("scaler", SklearnCompatibleScaler()),
        ("en", ElasticNet(max_iter=10000))
    ])
    en_params = {"en__alpha": np.logspace(-4, 2, 30),
                 "en__l1_ratio": np.linspace(0.1, 0.9, 9)}
    en_gs = GridSearchCV(en_pipe, param_grid=en_params,
                         cv=kfold, scoring="neg_root_mean_squared_error")
    en_gs.fit(X_train, y_train)
    print(f"  ElasticNet 最优 alpha: {en_gs.best_params_['en__alpha']:.4f}")
    print(f"  ElasticNet 最优 l1_ratio: {en_gs.best_params_['en__l1_ratio']:.4f}")
    print(f"  ElasticNet 最优 CV RMSE: {-en_gs.best_score_:.4f}")

    # 绘制 CV 误差 vs alpha 曲线
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ridge CV curve
    ridge_cv_results = ridge_gs.cv_results_
    ridge_means = -ridge_cv_results["mean_test_score"]
    axes[0].plot(alpha_grid, ridge_means, "b-o", markersize=3)
    axes[0].axvline(ridge_gs.best_params_["ridge__alpha"], color="r", ls="--",
                    label=f"Best alpha={ridge_gs.best_params_['ridge__alpha']:.3f}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("CV RMSE")
    axes[0].set_title("Ridge CV Error Curve")
    axes[0].legend()

    # Lasso CV curve
    lasso_cv_results = lasso_gs.cv_results_
    lasso_means = -lasso_cv_results["mean_test_score"]
    axes[1].plot(alpha_grid, lasso_means, "g-o", markersize=3)
    axes[1].axvline(lasso_gs.best_params_["lasso__alpha"], color="r", ls="--",
                    label=f"Best alpha={lasso_gs.best_params_['lasso__alpha']:.3f}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("CV RMSE")
    axes[1].set_title("Lasso CV Error Curve")
    axes[1].legend()

    # ElasticNet CV heatmap (简化: alpha vs l1_ratio)
    en_cv_results = en_gs.cv_results_
    en_scores = -en_cv_results["mean_test_score"].reshape(
        len(en_params["en__l1_ratio"]), len(en_params["en__alpha"]))
    im = axes[2].contourf(en_params["en__alpha"], en_params["en__l1_ratio"], en_scores, levels=20)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("alpha")
    axes[2].set_ylabel("l1_ratio")
    axes[2].set_title("ElasticNet CV Error Heatmap")
    fig.colorbar(im, ax=axes[2], label="CV RMSE")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cv_error_curves.png", dpi=150)
    plt.close(fig)
    print("  CV 误差曲线已保存: cv_error_curves.png")

    # ---- A3.4 模型性格大比拼 ----
    print("\n--- A3.4 模型性格大比拼 (测试集) ---")
    best_models = {
        "OLS": LinearRegression(),
        "Ridge (CV best)": ridge_gs.best_estimator_,
        "Lasso (CV best)": lasso_gs.best_estimator_,
        "ElasticNet (CV best)": en_gs.best_estimator_,
    }

    # OLS 不需要标准化
    ols_test = LinearRegression()
    ols_test.fit(X_train, y_train)

    for name, model in best_models.items():
        if name == "OLS":
            y_pred = ols_test.predict(X_test)
            coef_raw = ols_test.coef_
        else:
            y_pred = model.predict(X_test)
            # 从 Pipeline 获取系数并转回原始尺度
            pipe_scaler = model.named_steps["scaler"]
            std_arr = pipe_scaler._scaler.std_
            if name.startswith("Ridge"):
                coef_scaled = model.named_steps["ridge"].coef_
            elif name.startswith("Lasso"):
                coef_scaled = model.named_steps["lasso"].coef_
            else:
                coef_scaled = model.named_steps["en"].coef_
            coef_raw = coef_scaled / std_arr

        rmse = calculate_rmse(y_test, y_pred)
        mae  = calculate_mae(y_test, y_pred)
        print(f"\n  [{name}]")
        print(f"    RMSE: {rmse:.4f},  MAE: {mae:.4f}")
        print(f"    Coefficients:")
        for fn, cv in zip(feature_names, coef_raw):
            print(f"      {fn:<20s}: {cv:>10.4f}")

    # ---- A4. 传统变量选择 ----
    print("\n--- A4. 传统变量选择机制 ---")
    fs_selected, fs_scores = forward_selection(
        X_train, y_train, max_features=6, cv_folds=5, scoring="rmse", verbose=True)
    fs_selected_names = [feature_names[i] for i in fs_selected]

    be_selected, be_scores = backward_elimination(
        X_train, y_train, min_features=3, cv_folds=5, scoring="rmse", verbose=True)
    be_selected_names = [feature_names[i] for i in be_selected]

    # Lasso 选出的非零变量
    lasso_coef = lasso_gs.best_estimator_.named_steps["lasso"].coef_
    lasso_nonzero = [feature_names[i] for i, c in enumerate(lasso_coef) if abs(c) > 1e-8]
    print(f"\n  Lasso 非零变量: {lasso_nonzero}")

    # ---- 生成 synthetic_report.md ----
    print(f"\n  生成 {SYNTHETIC_MD} ...")
    generate_synthetic_report(
        ols_std, ridge_std, corr_labels,
        ridge_gs, lasso_gs, en_gs,
        best_models, feature_names, ols_test,
        X_test, y_test,
        fs_selected_names, be_selected_names, lasso_nonzero,
    )

    # ---- 生成 summary_comparison.md ----
    print(f"  生成 {SUMMARY_MD} ...")
    generate_summary_comparison()

    print("\nTask A 完成！")
    return lasso_nonzero, fs_selected_names, be_selected_names


# ============================================================
#                     Task B (housing.csv)
# ============================================================
def task_b():
    print("\n" + "=" * 70)
    print("Task B: housing.csv 真实数据 — 正则化与变量筛选")
    print("=" * 70)

    # ---- B1. 读取与说明 ----
    df = pd.read_csv(HOUSING_CSV)
    print(f"  数据形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")

    # 目标变量: MEDV (房价中位数)
    target_col = "MEDV"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    n, p = X.shape
    print(f"  特征数: {p} (>=8, 适宜正则化)")

    # 检查缺失值
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print("  检测到缺失值, 使用均值填充")
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="mean")
        X = imp.fit_transform(X)

    # ---- B2. 完整流程 ----
    print("\n--- B2: OLS vs Ridge vs Lasso vs ElasticNet ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 自定义 Scaler 包装
    from sklearn.base import BaseEstimator, TransformerMixin
    class SklearnCompatibleScaler(BaseEstimator, TransformerMixin):
        def __init__(self):
            self._scaler = CustomStandardScaler()
        def fit(self, X, y=None):
            self._scaler.fit(X)
            return self
        def transform(self, X, y=None):
            return self._scaler.transform(X)

    scaler_wrapper = SklearnCompatibleScaler()

    # OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)

    # GridSearch for Ridge, Lasso, ElasticNet
    alpha_grid = np.logspace(-4, 4, 50)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Ridge
    ridge_pipe = Pipeline([("scaler", SklearnCompatibleScaler()), ("ridge", Ridge())])
    ridge_gs = GridSearchCV(ridge_pipe, {"ridge__alpha": alpha_grid},
                            cv=kfold, scoring="neg_root_mean_squared_error")
    ridge_gs.fit(X_train, y_train)
    y_pred_ridge = ridge_gs.predict(X_test)
    # 转回原始尺度系数
    ridge_std_arr = ridge_gs.best_estimator_.named_steps["scaler"]._scaler.std_
    ridge_coef_raw = ridge_gs.best_estimator_.named_steps["ridge"].coef_ / ridge_std_arr

    # Lasso
    lasso_pipe = Pipeline([("scaler", SklearnCompatibleScaler()), ("lasso", Lasso(max_iter=10000))])
    lasso_gs = GridSearchCV(lasso_pipe, {"lasso__alpha": alpha_grid},
                            cv=kfold, scoring="neg_root_mean_squared_error")
    lasso_gs.fit(X_train, y_train)
    y_pred_lasso = lasso_gs.predict(X_test)
    lasso_std_arr = lasso_gs.best_estimator_.named_steps["scaler"]._scaler.std_
    lasso_coef_raw = lasso_gs.best_estimator_.named_steps["lasso"].coef_ / lasso_std_arr

    # ElasticNet
    en_pipe = Pipeline([("scaler", SklearnCompatibleScaler()), ("en", ElasticNet(max_iter=10000))])
    en_params = {"en__alpha": np.logspace(-4, 2, 20),
                 "en__l1_ratio": np.linspace(0.1, 0.9, 9)}
    en_gs = GridSearchCV(en_pipe, en_params, cv=kfold, scoring="neg_root_mean_squared_error")
    en_gs.fit(X_train, y_train)
    y_pred_en = en_gs.predict(X_test)
    en_std_arr = en_gs.best_estimator_.named_steps["scaler"]._scaler.std_
    en_coef_raw = en_gs.best_estimator_.named_steps["en"].coef_ / en_std_arr

    # ---- 评估汇总 ----
    print("\n  === 测试集表现 ===")
    def print_eval(name, y_true, y_pred):
        rmse = calculate_rmse(y_true, y_pred)
        mae  = calculate_mae(y_true, y_pred)
        print(f"  {name:<20s}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        return rmse, mae

    results = {}
    for name, yp in [("OLS", y_pred_ols), ("Ridge", y_pred_ridge),
                     ("Lasso", y_pred_lasso), ("ElasticNet", y_pred_en)]:
        results[name] = print_eval(name, y_test, yp)

    # ---- 特征重要度 ----
    print("\n  === 各模型系数 (原始尺度) ===")
    for model_name, coef_arr in [("OLS", ols.coef_), ("Ridge", ridge_coef_raw),
                                  ("Lasso", lasso_coef_raw), ("ElasticNet", en_coef_raw)]:
        print(f"\n  [{model_name}]")
        nonzero = []
        for fn, cv in sorted(zip(feature_cols, coef_arr), key=lambda x: -abs(x[1])):
            print(f"    {fn:<10s}: {cv:>12.6f}")
            if abs(cv) > 1e-8:
                nonzero.append(fn)
        print(f"    非零变量数: {len(nonzero)}")
        if model_name == "Lasso":
            lasso_kept = nonzero

    # Lasso 剔除的变量
    lasso_removed = [c for c in feature_cols if c not in lasso_kept]
    print(f"\n  Lasso 剔除的变量: {lasso_removed}")

    # ---- B3: 生成 kaggle_report.md ----
    print(f"\n  生成 {KAGGLE_MD} ...")
    # 获取最优超参数
    best_ridge_alpha = ridge_gs.best_params_["ridge__alpha"]
    best_lasso_alpha = lasso_gs.best_params_["lasso__alpha"]
    best_en_alpha = en_gs.best_params_["en__alpha"]
    best_en_l1 = en_gs.best_params_["en__l1_ratio"]

    generate_kaggle_report(
        feature_cols, target_col, results,
        ols.coef_, ridge_coef_raw, lasso_coef_raw, en_coef_raw,
        lasso_kept, lasso_removed,
        best_ridge_alpha, best_lasso_alpha, best_en_alpha, best_en_l1,
    )

    print("\nTask B 完成！")
    return results


# ============================================================
#                   报告生成函数
# ============================================================
def generate_synthetic_report(
    ols_std, ridge_std, corr_labels,
    ridge_gs, lasso_gs, en_gs,
    best_models, feature_names, ols_test,
    X_test, y_test,
    fs_names, be_names, lasso_names,
):
    """生成 synthetic_report.md"""
    lines = []

    def w(s=""):
        lines.append(s)

    w("# Synthetic Correlated Data Report")
    w()
    w("## DGP (Data Generating Process)")
    w()
    w("**真实公式:**")
    w("```")
    w("y = 3.0 + 2.0 * X0 + 1.5 * X3 + ε")
    w("ε ~ N(0, 0.5²)")
    w("```")
    w()
    w("### 特征说明")
    w()
    w("| 特征 | 描述 | 是否真实信号 |")
    w("|------|------|-------------|")
    w("| X0_true   | 均匀分布 [0, 10] | ✅ 系数=2.0 |")
    w("| X1_corr+  | X0*2 + noise     | ❌ 与 X0 高度正相关 (r≈0.97) |")
    w("| X2_corr-  | X0*(-0.8)+noise  | ❌ 与 X0 高度负相关 (r≈-0.9) |")
    w("| X3_true   | N(0,1)           | ✅ 系数=1.5 |")
    w("| X4_noise  | N(0,1)           | ❌ 纯噪声 |")
    w("| X5_noise  | N(0,1)           | ❌ 纯噪声 |")
    w("| X6_noise  | N(0,1)           | ❌ 纯噪声 |")
    w("| X7_corr+  | X3*1.5+noise     | ❌ 与 X3 高度正相关 (r≈0.98) |")
    w("| X8_noise  | N(0,1)           | ❌ 纯噪声 |")
    w()
    w("**共线性族:** ")
    w("- 族1: X0, X1, X2 (高度相关)")
    w("- 族2: X3, X7 (高度相关)")
    w("- 噪声: X4, X5, X6, X8")
    w()

    w("## A3.1 稳定性对比 (50 次随机切分)")
    w()
    w("### 系数标准差 (仅展示高度相关族)")
    w()
    w("| 特征 | OLS std | Ridge std | 稳定性改善 |")
    w("|------|---------|-----------|-----------|")
    for name, os_, rs_ in zip(corr_labels, ols_std, ridge_std):
        improvement = "✅" if rs_ < os_ else "—"
        w(f"| {name} | {os_:.4f} | {rs_:.4f} | {improvement} |")
    w()
    w("![Stability Boxplot](stability_boxplot.png)")
    w()

    w("## A3.2 Pipeline 与标准化")
    w()
    w("**为什么 Ridge/Lasso 前必须标准化?**")
    w()
    w("Ridge 和 Lasso 的罚项是对系数向量施加 L2/L1 约束。")
    w("如果特征尺度不一致（如 X0 范围 [0,10]，X3 范围 [-3,3])，")
    w("则罚项对不同特征的惩罚力度实际不同——尺度大的特征会被\"过度惩罚\"。")
    w("标准化后每个特征均值=0、标准差=1，罚项对所有特征公平施加。")
    w()

    w("## A3.3 GridSearchCV 最优超参数")
    w()
    w(f"- **Ridge** 最优 alpha = `{ridge_gs.best_params_['ridge__alpha']:.4f}`")
    w(f"- **Lasso** 最优 alpha = `{lasso_gs.best_params_['lasso__alpha']:.4f}`")
    w(f"- **ElasticNet** 最优 alpha = `{en_gs.best_params_['en__alpha']:.4f}`, "
      f"l1_ratio = `{en_gs.best_params_['en__l1_ratio']:.4f}`")
    w()
    w("![CV Error Curves](cv_error_curves.png)")
    w()

    w("## A3.4 模型性格大比拼 (测试集)")
    w()
    w("### 测试集表现")
    w()
    w("| 模型 | RMSE | MAE |")
    w("|------|------|-----|")
    for name, model in best_models.items():
        if name == "OLS":
            yp = ols_test.predict(X_test)
        else:
            yp = model.predict(X_test)
        rmse = calculate_rmse(y_test, yp)
        mae = calculate_mae(y_test, yp)
        w(f"| {name} | {rmse:.4f} | {mae:.4f} |")
    w()

    w("### 各模型系数对比")
    w()
    w("| 特征 | OLS | Ridge | Lasso | ElasticNet |")
    w("|------|-----|-------|-------|------------|")
    for i, fn in enumerate(feature_names):
        ols_c = ols_test.coef_[i]
        ridge_c = ridge_gs.best_estimator_.named_steps["ridge"].coef_[i] / \
                  ridge_gs.best_estimator_.named_steps["scaler"]._scaler.std_[i]
        lasso_c = lasso_gs.best_estimator_.named_steps["lasso"].coef_[i] / \
                  lasso_gs.best_estimator_.named_steps["scaler"]._scaler.std_[i]
        en_step = en_gs.best_estimator_.named_steps["en"]
        en_scaler = en_gs.best_estimator_.named_steps["scaler"]
        en_c = en_step.coef_[i] / en_scaler._scaler.std_[i]
        w(f"| {fn} | {ols_c:.4f} | {ridge_c:.4f} | {lasso_c:.4f} | {en_c:.4f} |")
    w()

    w("### 模型性格分析")
    w()
    w("- **Ridge:** 对共线性族 X0/X1/X2 和 X3/X7 均进行均匀缩小，不会将系数压缩至零。")
    w("  所有特征都保留，只是在幅度上受到约束。")
    w("- **Lasso:** 展现出\"赢者通吃\"行为。在共线性族中可能只保留一个（如 X0），")
    w("  而将其相关变量 X1/X2 压缩至零。这是 L1 惩罚的固有性质。")
    w("- **ElasticNet:** 介于两者之间。l1_ratio 控制 L1 和 L2 的比例，")
    w("  既能像 Lasso 一样做变量选择，又能像 Ridge 一样保留共线组的部分结构。")
    w()

    w("## A4. 变量筛选对比")
    w()
    w(f"- **Forward Selection (Top-6):** {fs_names}")
    w(f"- **Backward Elimination:** {be_names}")
    w(f"- **Lasso 非零变量:** {lasso_names}")
    w()
    w("### 一致性分析")
    w()
    w("传统的逐步回归（前向/后向）与 Lasso 在变量选择上可能存在差异：")
    w("- Lasso 通过连续惩罚实现变量选择，效率更高")
    w("- 前向选择每次只加一个变量，可能忽略变量间的联合效应")
    w("- 后向剔除从全模型开始，计算开销大但对变量间关系考虑更全面")
    w()

    SYNTHETIC_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {SYNTHETIC_MD} 已生成")


def generate_kaggle_report(
    feature_cols, target_col, results,
    ols_coef, ridge_coef, lasso_coef, en_coef,
    lasso_kept, lasso_removed,
    best_ridge_alpha, best_lasso_alpha, best_en_alpha, best_en_l1,
):
    """生成 kaggle_report.md"""
    lines = []

    def w(s=""):
        lines.append(s)

    w("# Kaggle Housing Data Report")
    w()
    w("## 数据来源与业务背景")
    w()
    w("- **数据集:** Boston Housing (housing.csv)")
    w("- **目标:** 预测房价中位数 (MEDV)")
    w("- **特征数:** 13 个 (含连续与分类)")
    w("- **样本数:** ~506")
    w()
    w("该数据集存在一定共线性（如 RAD/TAX, NOX/INDUS 等），")
    w("适合用于验证正则化方法在真实场景下的表现。")
    w()

    w("## 测试集表现")
    w()
    w("| 模型 | RMSE | MAE |")
    w("|------|------|-----|")
    for name, (rmse, mae) in results.items():
        w(f"| {name} | {rmse:.4f} | {mae:.4f} |")
    w()

    w("## 最优超参数")
    w()
    w(f"- Ridge α = {best_ridge_alpha:.4f}")
    w(f"- Lasso α = {best_lasso_alpha:.4f}")
    w(f"- ElasticNet α = {best_en_alpha:.4f}, l1_ratio = {best_en_l1:.4f}")
    w()

    w("## 各模型系数 (原始尺度)")
    w()
    w("| 特征 | OLS | Ridge | Lasso | ElasticNet |")
    w("|------|-----|-------|-------|------------|")
    for i, fn in enumerate(feature_cols):
        w(f"| {fn} | {ols_coef[i]:.4f} | {ridge_coef[i]:.4f} | "
          f"{lasso_coef[i]:.4f} | {en_coef[i]:.4f} |")
    w()

    w("## Lasso 变量选择")
    w()
    w(f"**保留 ({len(lasso_kept)} 个):** {lasso_kept}")
    w(f"**剔除 ({len(lasso_removed)} 个):** {lasso_removed}")
    w()

    w("## B3. 真实数据推测解释")
    w()
    w("### Q1: 与 OLS 相比，正则化方法是否显著提升了验证集表现？")
    w()
    ols_rmse = results["OLS"][0]
    ridge_rmse = results["Ridge"][0]
    lasso_rmse = results["Lasso"][0]
    en_rmse = results["ElasticNet"][0]

    if ridge_rmse < ols_rmse or lasso_rmse < ols_rmse:
        w("正则化方法在测试集上展现了更优或相近的 RMSE。")
        w("特别是在特征数较多且存在共线性的情况下，Ridge/Lasso 通过偏差-方差权衡，")
        w("牺牲少量偏差换来更低的方差，从而在测试集上表现更好。")
    else:
        w("如果正则化并未显著提升验证集表现，可能原因：")
        w("- 数据本身的共线性不够严重，OLS 已经接近最优")
        w("- 样本量相对特征数足够大，OLS 方差可控")
        w("- 正则化强度选择偏保守")
    w(f"  (OLS RMSE={ols_rmse:.4f} vs Ridge={ridge_rmse:.4f} vs Lasso={lasso_rmse:.4f})")
    w()

    w("### Q2: Lasso 剔除了哪些特征？合理吗？")
    w()
    w(f"Lasso 剔除: {lasso_removed}")
    w("从业务逻辑看：")
    w("- 若被剔除的特征与其他保留特征高度相关（如 INDUS 与 NOX），")
    w("  则 Lasso 选择保留其中一个而剔除另一个是合理的")
    w("- 若被剔除的特征在业务上确实影响房价，可能是由于该特征的信息")
    w("  已被其他特征替代，或被交叉验证判定为增加噪声")
    w()

    w("### Q3: 最关键的 5 个影响因素")
    w()
    # 按 ElasticNet 系数绝对值排序
    en_importance = sorted(zip(feature_cols, en_coef), key=lambda x: -abs(x[1]))
    top5 = [f"{name}({coef:.4f})" for name, coef in en_importance[:5]]
    w(f"以 **ElasticNet** 的结果为准, Top-5: {top5}")
    w()
    w("理由：ElasticNet 兼顾了 L1 的变量选择能力和 L2 对共线性族的稳定处理，")
    w("在真实数据场景下更为鲁棒。既不像 Lasso 那样过度激进（丢掉有用变量），")
    w("也不像 Ridge 那样保留所有变量（失去解释简洁性）。")
    w()

    KAGGLE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {KAGGLE_MD} 已生成")


def generate_summary_comparison():
    """生成 summary_comparison.md"""
    lines = []

    def w(s=""):
        lines.append(s)

    w("# 理论与实践总结 (Summary & Comparison)")
    w()
    w("## Q1: Lasso 在处理高度相关变量组时的风险 & ElasticNet 如何缓解")
    w()
    w("**风险：** Lasso 在高度相关的特征族中倾向于随机保留其中一个而剔除其余。")
    w("在业务场景下，如果被剔除的变量在因果上很重要但统计上与保留者高度相关，")
    w("那么模型给出的\"关键因素\"名单可能具有误导性。")
    w("此外，轻微的数据扰动可能导致 Lasso 选中同一族中不同的代表变量，")
    w("结论稳定性不足。")
    w()
    w("**ElasticNet 的缓解：** 通过混合 L1 和 L2 惩罚（由 l1_ratio 调节），")
    w("ElasticNet 在变量选择时倾向于把共线组的系数一起缩小而非二选一。")
    w("这既能保留 Ridge 的\"组团收缩\"稳定性，又能实现一定程度的稀疏性。")
    w()

    w("## Q2: GridSearchCV 最优 vs \"越稀疏越好\" vs \"越稳越好\"")
    w()
    w("- **GridSearchCV:** 以交叉验证误差最小化为目标。自动找到预测性能最优的 α。")
    w("- **越稀疏越好:** 主观追求更简洁的模型（少数几个系数非零）。")
    w("  但这可能在 CV 误差上并非最优——去掉太多变量可能损失预测力。")
    w("- **越稳越好:** 追求系数在面对不同样本时方差最小。大 α 的 Ridge 最稳定，")
    w("  但可能偏差过大导致欠拟合。")
    w()
    w("三者代表了模型选择中 **偏差-方差-稀疏性** 的三角权衡。")
    w("GridSearchCV 是数据驱动的折中方案，而\"越稀疏/越稳\"是主观偏好。")
    w()

    w("## Q3: 传统变量选择 vs Lasso — 计算效率与结果体会")
    w()
    w("**计算效率:**")
    w("- 前向选择/后向剔除每步需要多次拟合 OLS 和交叉验证，复杂度为 O(p² × n × cv)")
    w("- Lasso 通过坐标下降法高效求解整个正则化路径，且一次拟合即可得到稀疏解")
    w("- 当特征数 p 很大时，Lasso 的效率优势显著")
    w()
    w("**最终结果:**")
    w("- 传统逐步回归是离散的（要么选要么不选），Lasso 是连续的（系数逐渐收缩至零）")
    w("- 逐步回归受变量进入顺序影响较大，Lasso 通过全局优化避免了这个问题")
    w("- 在共线性场景下 Lasso 选出的变量可能不如 Ridge/ElasticNet 稳定")
    w()
    w("**体会:** 没有一种方法在所有场景下最优。建议：")
    w("1. 用 Lasso 快速筛选候选变量")
    w("2. 用 Ridge/ElasticNet 在候选集上稳定估计")
    w("3. 结合业务知识做最终判断")
    w()

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {SUMMARY_MD} 已生成")


# ============================================================
#                          主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Week 13: Regularized Regression & Variable Selection")
    print("=" * 70)

    task_a()
    task_b()

    print("\n" + "=" * 70)
    print("All tasks completed. 查看 results/ 目录下的报告。")
    print("=" * 70)
