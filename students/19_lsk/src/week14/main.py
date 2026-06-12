#!/usr/bin/env python3
"""
Week 14: High-Dimensional Regression, PCA, and PCR
==================================================
Task A: 高维共线性如何破坏 OLS → 虚假低训练误差 + 系数不稳定
Task B: PCA → PCR 工作流（先压缩再回归）
Task C: Lasso vs PCR — 变量筛选 vs 信息压缩
Task D: (optional) 真实数据挑战

单一入口: uv run src/week14/main.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ── 复用 utils ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.transformers import CustomStandardScaler
from src.utils.metrics import calculate_rmse, calculate_mae
from src.utils.models import (
    AnalyticalOLS, PCR, coefficient_stability,
    make_high_dimensional_data, make_sparse_regression_data,
)

# ── 路径 ──
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_CSV    = DATA_DIR / "synthetic_highdim.csv"
SYNTHETIC_SPARSE  = DATA_DIR / "synthetic_sparse.csv"
HOUSING_CSV      = DATA_DIR / "housing.csv"

SYNTHETIC_MD = RESULTS_DIR / "synthetic_report.md"
SUMMARY_MD   = RESULTS_DIR / "summary_comparison.md"
KAGGLE_MD    = RESULTS_DIR / "kaggle_report.md"

# ══════════════════════════════════════════════════════════════
#  Task A: 高维共线性 → OLS 失败演示
# ══════════════════════════════════════════════════════════════
def task_a():
    print("\n" + "=" * 70)
    print("Task A: 高维数据 → OLS 失败演示")
    print("=" * 70)

    # ── A1. 生成高维低秩数据 ──
    print("\n--- A1: 生成高维低秩模拟数据 ---")
    X_full, y_full, F = make_high_dimensional_data(
        n_samples=150, n_features=80, n_latent_factors=5,
        noise_std=0.5, seed=42,
    )
    n_total, p_total = X_full.shape
    print(f"  样本数 n={n_total}, 特征数 p={p_total}, 潜在因子数=5")

    # ── A2. 保存 CSV ──
    col_names = [f"X{i}" for i in range(p_total)]
    df = pd.DataFrame(X_full, columns=col_names)
    df["y"] = y_full
    df.to_csv(SYNTHETIC_CSV, index=False)
    print(f"  数据已保存: {SYNTHETIC_CSV}")

    # ── A3. 不同 p 下 OLS 的 train/test RMSE + 矩阵结构 ──
    print("\n--- A3: 不同特征维度下 OLS 表现 ---")
    p_list = [10, 30, 60, 80, 100]
    # 确保 p 不超过生成的总特征数
    p_list = [p for p in p_list if p <= p_total]

    records = []
    for p_val in p_list:
        X_sub = X_full[:120, :p_val]   # 固定 120 训练样本
        y_sub = y_full[:120]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.3, random_state=42)

        # 矩阵诊断
        rank = np.linalg.matrix_rank(X_train)
        # condition number via SVD
        try:
            _, S, _ = np.linalg.svd(X_train, full_matrices=False)
            cond_num = S[0] / S[-1] if S[-1] > 1e-12 else np.inf
        except np.linalg.LinAlgError:
            cond_num = np.inf

        # OLS
        X_train_c = np.column_stack([np.ones(len(y_train)), X_train])
        X_test_c  = np.column_stack([np.ones(len(y_test)),  X_test])
        try:
            ols = AnalyticalOLS()
            ols.fit(X_train_c, y_train)
            train_rmse = calculate_rmse(y_train, ols.predict(X_train_c))
            test_rmse  = calculate_rmse(y_test,  ols.predict(X_test_c))
        except np.linalg.LinAlgError:
            train_rmse = np.nan
            test_rmse  = np.nan

        records.append({
            "p": p_val, "rank": rank, "cond_num": cond_num,
            "train_rmse": train_rmse, "test_rmse": test_rmse,
        })
        print(f"  p={p_val:>4d}: rank={rank:>4d}, cond_num={cond_num:>12.2e}, "
              f"train_rmse={train_rmse:.4f}, test_rmse={test_rmse:.4f}")

    # 图 1: RMSE vs p
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    p_vals = [r["p"] for r in records]
    axes[0].plot(p_vals, [r["train_rmse"] for r in records], "bo-", label="Train RMSE")
    axes[0].plot(p_vals, [r["test_rmse"]  for r in records], "ro-", label="Test RMSE")
    axes[0].set_xlabel("Number of features p")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("OLS Train vs Test RMSE (n_train=84 fixed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 图 2: Rank & Condition Number vs p
    ax2 = axes[1]
    ax2.bar(np.array(p_vals) - 1.5, [r["rank"] for r in records],
            width=3, label="rank(X_train)", color="steelblue", alpha=0.7)
    ax2.set_xlabel("Number of features p")
    ax2.set_ylabel("Rank", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(p_vals, [r["cond_num"] for r in records], "rs-", markersize=6,
                  label="Condition Number")
    ax2_twin.set_ylabel("Condition Number (log scale)", color="red")
    ax2_twin.set_yscale("log")
    ax2_twin.tick_params(axis="y", labelcolor="red")
    ax2.set_title("Matrix Structure vs p")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "task_a_rmse_and_matrix.png", dpi=150)
    plt.close(fig)
    print("  图已保存: task_a_rmse_and_matrix.png")

    # ── A4. 系数稳定性：50 次随机切分 ──
    print("\n--- A4: OLS 系数稳定性 (50 splits, p=60) ---")
    X_a4, y_a4, _ = make_high_dimensional_data(
        n_samples=150, n_features=60, n_latent_factors=5,
        noise_std=0.5, seed=42,
    )
    n_splits = 50
    # 选 6 个代表性变量（前 3 个和第 30, 31, 59）
    key_vars = [0, 1, 2, 10, 30, 59]
    key_names = [f"X{i}" for i in key_vars]

    coefs_all = np.zeros((n_splits, len(key_vars)))
    for split_i in range(n_splits):
        X_tr, _, y_tr, _ = train_test_split(
            X_a4, y_a4, test_size=0.3, random_state=split_i)
        X_tr_c = np.column_stack([np.ones(len(y_tr)), X_tr])
        try:
            ols = AnalyticalOLS()
            ols.fit(X_tr_c, y_tr)
            coefs_all[split_i, :] = ols.coef_[1:][key_vars]
        except np.linalg.LinAlgError:
            coefs_all[split_i, :] = np.nan

    # 箱线图
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot([coefs_all[:, i] for i in range(len(key_vars))],
                    tick_labels=key_names, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.set_title(f"OLS Coefficient Instability Across {n_splits} Splits (p=60, n=105 train)")
    ax.set_ylabel("Coefficient Value")
    ax.set_xlabel("Feature")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "task_a_coef_instability.png", dpi=150)
    plt.close(fig)
    print("  图已保存: task_a_coef_instability.png")

    # 打印标准差
    coef_stds = np.nanstd(coefs_all, axis=0)
    print("\n  --- 系数标准差 (越低越稳定) ---")
    for nm, s in zip(key_names, coef_stds):
        print(f"  {nm}: std={s:.4f}")

    return records, p_list


# ══════════════════════════════════════════════════════════════
#  Task B: PCA → PCR
# ══════════════════════════════════════════════════════════════
def task_b():
    print("\n" + "=" * 70)
    print("Task B: PCA → PCR 工作流")
    print("=" * 70)

    # 用同一份数据
    X, y, F = make_high_dimensional_data(
        n_samples=150, n_features=80, n_latent_factors=5,
        noise_std=0.5, seed=42,
    )
    p_total = X.shape[1]
    print(f"  数据: n={len(y)}, p={p_total}, latent_factors=5")

    # ── B1. PCA 累计解释方差 ──
    print("\n--- B1: PCA 累计解释方差 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # 找 80%, 90%, 95% 对应 PC 数
    for thresh in [0.8, 0.9, 0.95]:
        k_thresh = np.argmax(cum_var >= thresh) + 1
        print(f"  累计方差 ≥ {thresh*100:.0f}%: k = {k_thresh}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum_var)+1), cum_var, "b-", linewidth=2)
    ax.axhline(y=0.8, color="orange", ls="--", label="80%")
    ax.axhline(y=0.9, color="green", ls="--", label="90%")
    ax.axhline(y=0.95, color="red", ls="--", label="95%")
    ax.set_xlabel("Number of Principal Components (k)")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "task_b_pca_variance.png", dpi=150)
    plt.close(fig)
    print("  图已保存: task_b_pca_variance.png")

    # ── B2. PCR 不同 k ──
    print("\n--- B2: PCR 在不同 k 下的表现 ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    max_k = min(30, X_train.shape[1], X_train.shape[0])
    k_list = list(range(1, max_k + 1))
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    train_rmses, test_rmses, cv_rmses = [], [], []

    for k in k_list:
        pcr = PCR(n_components=k)
        pcr.fit(X_train, y_train)
        train_rmses.append(calculate_rmse(y_train, pcr.predict(X_train)))
        test_rmses.append(calculate_rmse(y_test, pcr.predict(X_test)))

        # 5-fold CV
        cv_scores = []
        for tr_idx, val_idx in kfold.split(X_train):
            X_tr_fold, X_val_fold = X_train[tr_idx], X_train[val_idx]
            y_tr_fold, y_val_fold = y_train[tr_idx], y_train[val_idx]
            pcr_cv = PCR(n_components=k)
            pcr_cv.fit(X_tr_fold, y_tr_fold)
            cv_scores.append(calculate_rmse(y_val_fold, pcr_cv.predict(X_val_fold)))
        cv_rmses.append(np.mean(cv_scores))

    # OLS 基线 (使用 sklearn LinearRegression，因为它能处理奇异矩阵)
    ols_sk = LinearRegression()
    ols_sk.fit(X_train, y_train)
    ols_train_rmse = calculate_rmse(y_train, ols_sk.predict(X_train))
    ols_test_rmse  = calculate_rmse(y_test,  ols_sk.predict(X_test))

    best_k_idx = np.argmin(cv_rmses)
    best_k = k_list[best_k_idx]
    print(f"  最优 k (min CV RMSE): {best_k}, CV RMSE={cv_rmses[best_k_idx]:.4f}")
    print(f"  OLS 基线: train_rmse={ols_train_rmse:.4f}, test_rmse={ols_test_rmse:.4f}")

    # 图: train/test/CV vs k
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_list, train_rmses, "b-", label="PCR Train RMSE", alpha=0.7)
    ax.plot(k_list, test_rmses,  "r-", label="PCR Test RMSE",  alpha=0.7)
    ax.plot(k_list, cv_rmses,    "g-", label="PCR CV RMSE (5-fold)", alpha=0.7)
    ax.axhline(y=ols_test_rmse, color="orange", ls="--",
               label=f"OLS Test RMSE = {ols_test_rmse:.4f}")
    ax.axvline(x=best_k, color="purple", ls=":", label=f"Best k={best_k}")
    ax.set_xlabel("Number of Principal Components (k)")
    ax.set_ylabel("RMSE")
    ax.set_title("PCR: Train / Test / CV RMSE vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "task_b_pcr_curves.png", dpi=150)
    plt.close(fig)
    print("  图已保存: task_b_pcr_curves.png")

    return {
        "cum_var": cum_var, "k_list": k_list,
        "train_rmses": train_rmses, "test_rmses": test_rmses,
        "cv_rmses": cv_rmses, "best_k": best_k,
        "ols_train_rmse": ols_train_rmse, "ols_test_rmse": ols_test_rmse,
    }


# ══════════════════════════════════════════════════════════════
#  Task C: Lasso vs PCR — Selection vs Compression
# ══════════════════════════════════════════════════════════════
def task_c():
    print("\n" + "=" * 70)
    print("Task C: Lasso vs PCR — Selection vs Compression")
    print("=" * 70)

    # ── C1. 构造两种数据世界 ──
    print("\n--- C1: 构造两种数据世界 ---")
    n_samples = 150
    n_features = 80

    # World 1: Sparse truth (只有 5 个原始变量驱动 y)
    X_sp, y_sp, true_coef_sp = make_sparse_regression_data(
        n_samples=n_samples, n_features=n_features,
        n_true_features=5, noise_std=0.5, seed=42)
    df_sp = pd.DataFrame(X_sp, columns=[f"X{i}" for i in range(n_features)])
    df_sp["y"] = y_sp
    df_sp.to_csv(SYNTHETIC_SPARSE, index=False)
    print(f"  Sparse数据已保存: {SYNTHETIC_SPARSE}")
    nz_sp = np.sum(np.abs(true_coef_sp) > 1e-8)
    print(f"  World 1 (Sparse): 仅 {nz_sp} 个变量有真实信号, 其余为噪声")

    # World 2: Latent-factor truth (5 个潜在因子驱动 80 个观测变量及 y)
    X_lf, y_lf, F_lf = make_high_dimensional_data(
        n_samples=n_samples, n_features=n_features,
        n_latent_factors=5, noise_std=0.5, seed=42)
    print(f"  World 2 (Latent-factor): 5 个潜在因子生成 80 个观测变量")

    # ── C2. Lasso vs PCR in each world ──
    print("\n--- C2: Lasso vs PCR 对比 ---")
    results = {}

    for world_name, X_w, y_w in [
        ("Sparse Truth", X_sp, y_sp),
        ("Latent-Factor Truth", X_lf, y_lf),
    ]:
        print(f"\n  === {world_name} ===")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_w, y_w, test_size=0.3, random_state=42)

        # ── Lasso (CV 自动选 α) ──
        lasso_cv = LassoCV(cv=5, max_iter=200000, random_state=42,
                           alphas=np.logspace(-4, 2, 50))
        lasso_cv.fit(X_tr, y_tr)
        y_pred_lasso = lasso_cv.predict(X_te)
        lasso_rmse = calculate_rmse(y_te, y_pred_lasso)
        lasso_mae  = calculate_mae(y_te, y_pred_lasso)
        lasso_nz   = np.sum(np.abs(lasso_cv.coef_) > 1e-8)
        lasso_alpha_opt = lasso_cv.alpha_

        # ── PCR (CV 选 k) ──
        max_k_pcr = min(30, X_tr.shape[1], X_tr.shape[0])
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        pcr_cv_rmses = []
        for k in range(1, max_k_pcr + 1):
            cv_scores = []
            for tr_i, val_i in kfold.split(X_tr):
                pcr_cv = PCR(n_components=k)
                pcr_cv.fit(X_tr[tr_i], y_tr[tr_i])
                cv_scores.append(calculate_rmse(y_tr[val_i], pcr_cv.predict(X_tr[val_i])))
            pcr_cv_rmses.append(np.mean(cv_scores))
        best_k_pcr = np.argmin(pcr_cv_rmses) + 1

        pcr_best = PCR(n_components=best_k_pcr)
        pcr_best.fit(X_tr, y_tr)
        y_pred_pcr = pcr_best.predict(X_te)
        pcr_rmse = calculate_rmse(y_te, y_pred_pcr)
        pcr_mae  = calculate_mae(y_te, y_pred_pcr)

        # ── OLS 基线 ──
        ols_sk = LinearRegression()
        ols_sk.fit(X_tr, y_tr)
        ols_rmse = calculate_rmse(y_te, ols_sk.predict(X_te))
        ols_mae  = calculate_mae(y_te, ols_sk.predict(X_te))

        print(f"    OLS:              RMSE={ols_rmse:.4f}, MAE={ols_mae:.4f}")
        print(f"    Lasso (α={lasso_alpha_opt:.4f}): RMSE={lasso_rmse:.4f}, MAE={lasso_mae:.4f}, "
              f"非零系数={lasso_nz}/{n_features}")
        print(f"    PCR   (k={best_k_pcr}):       RMSE={pcr_rmse:.4f}, MAE={pcr_mae:.4f}, "
              f"主成分数={best_k_pcr}")

        # 稳定性
        try:
            lasso_stab = coefficient_stability(
                X_tr, y_tr, n_splits=30, test_size=0.2,
                model_factory=lambda: Lasso(alpha=lasso_alpha_opt, max_iter=10000))
        except Exception:
            lasso_stab = {"stability_score": np.nan, "coefs_mean": [], "coefs_std": []}

        try:
            pcr_stab = coefficient_stability(
                X_tr, y_tr, n_splits=30, test_size=0.2,
                model_factory=lambda: PCR(n_components=best_k_pcr))
        except Exception:
            pcr_stab = {"stability_score": np.nan, "coefs_mean": [], "coefs_std": []}

        results[world_name] = {
            "OLS": {"rmse": ols_rmse, "mae": ols_mae},
            "Lasso": {"rmse": lasso_rmse, "mae": lasso_mae, "n_nonzero": lasso_nz,
                      "alpha": lasso_alpha_opt, "stability": lasso_stab.get("stability_score", np.nan)},
            "PCR": {"rmse": pcr_rmse, "mae": pcr_mae, "k": best_k_pcr,
                    "stability": pcr_stab.get("stability_score", np.nan)},
        }

    # 图: 两场景并排柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_i, (world_name, res) in enumerate(results.items()):
        ax = axes[ax_i]
        methods = ["OLS", "Lasso", "PCR"]
        rmses = [res[m]["rmse"] for m in methods]
        colors = ["gray", "steelblue", "darkorange"]
        bars = ax.bar(methods, rmses, color=colors, alpha=0.8)
        for bar, v in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.4f}", ha="center", fontsize=9)
        ax.set_title(f"{world_name}\nTest RMSE")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "task_c_lasso_vs_pcr.png", dpi=150)
    plt.close(fig)
    print("\n  图已保存: task_c_lasso_vs_pcr.png")

    return results


# ══════════════════════════════════════════════════════════════
#  Task D (Optional): 真实数据 — housing.csv
# ══════════════════════════════════════════════════════════════
def task_d():
    print("\n" + "=" * 70)
    print("Task D (Optional): housing.csv 真实数据挑战")
    print("=" * 70)

    # 检查数据是否存在
    housing_paths = [
        HOUSING_CSV,
        Path(__file__).resolve().parents[1] / "week11" / "data" / "housing.csv",
    ]
    housing_path = None
    for hp in housing_paths:
        if hp.exists():
            housing_path = hp
            break

    if housing_path is None:
        print("  ⚠ housing.csv 未找到，跳过 Task D")
        return None

    df = pd.read_csv(housing_path)
    print(f"  数据形状: {df.shape}, 来源: {housing_path}")

    # 检测数据集类型（Boston 还是 California）
    if "MEDV" in df.columns:
        # Boston Housing
        target_col = "MEDV"
    elif "median_house_value" in df.columns:
        # California Housing
        target_col = "median_house_value"
    else:
        # 尝试最后一列作为目标
        target_col = df.columns[-1]
        print(f"  ⚠ 未识别目标列，使用最后一列: {target_col}")

    # 处理分类变量
    for col in df.columns:
        if df[col].dtype == object and col != target_col:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    # 检查并处理缺失值
    if np.any(np.isnan(X)):
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy="mean").fit_transform(X)
    if np.any(np.isnan(y)):
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

    n, p = X.shape
    print(f"  样本数 n={n}, 特征数 p={p}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # ── OLS ──
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_rmse = calculate_rmse(y_test, ols.predict(X_test))
    ols_mae  = calculate_mae(y_test, ols.predict(X_test))

    # ── Lasso ──
    lasso_cv = LassoCV(cv=5, max_iter=200000, random_state=42,
                       alphas=np.logspace(-4, 2, 50))
    lasso_cv.fit(X_train, y_train)
    lasso_rmse = calculate_rmse(y_test, lasso_cv.predict(X_test))
    lasso_mae  = calculate_mae(y_test, lasso_cv.predict(X_test))
    lasso_nz   = np.sum(np.abs(lasso_cv.coef_) > 1e-8)

    # ── PCR ──
    max_k = min(20, p)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    pcr_cv_rmses = []
    for k in range(1, max_k + 1):
        cv_scores = []
        for tr_i, val_i in kfold.split(X_train):
            pcr_cv = PCR(n_components=k)
            pcr_cv.fit(X_train[tr_i], y_train[tr_i])
            cv_scores.append(calculate_rmse(y_train[val_i],
                                             pcr_cv.predict(X_train[val_i])))
        pcr_cv_rmses.append(np.mean(cv_scores))
    best_k = np.argmin(pcr_cv_rmses) + 1
    pcr_best = PCR(n_components=best_k)
    pcr_best.fit(X_train, y_train)
    pcr_rmse = calculate_rmse(y_test, pcr_best.predict(X_test))
    pcr_mae  = calculate_mae(y_test, pcr_best.predict(X_test))

    # 诊断
    rank_train = np.linalg.matrix_rank(X_train)
    try:
        _, S, _ = np.linalg.svd(X_train, full_matrices=False)
        cond_num = S[0] / S[-1] if S[-1] > 1e-12 else np.inf
    except Exception:
        cond_num = np.inf

    print(f"\n  === 测试集表现 ===")
    print(f"  OLS:              RMSE={ols_rmse:.4f}, MAE={ols_mae:.4f}")
    print(f"  Lasso (α={lasso_cv.alpha_:.4f}): RMSE={lasso_rmse:.4f}, MAE={lasso_mae:.4f}, "
          f"非零={lasso_nz}/{p}")
    print(f"  PCR   (k={best_k}):       RMSE={pcr_rmse:.4f}, MAE={pcr_mae:.4f}")
    print(f"  rank(X_train)={rank_train}, cond_num={cond_num:.2e}")

    # 判断更像 sparse 还是 latent-factor
    pca_full = PCA().fit(StandardScaler().fit_transform(X_train))
    cum_var_5 = np.sum(pca_full.explained_variance_ratio_[:5])
    print(f"  前 5 个 PC 解释方差: {cum_var_5*100:.1f}%")

    return {
        "OLS": {"rmse": ols_rmse, "mae": ols_mae},
        "Lasso": {"rmse": lasso_rmse, "mae": lasso_mae, "n_nonzero": lasso_nz,
                  "alpha": lasso_cv.alpha_, "coef": lasso_cv.coef_},
        "PCR": {"rmse": pcr_rmse, "mae": pcr_mae, "k": best_k},
        "rank": rank_train, "cond_num": cond_num,
        "cum_var_5": cum_var_5, "feature_cols": feature_cols,
        "n_samples": n, "n_features": p, "n_features_used": X_train.shape[1],
    }


# ══════════════════════════════════════════════════════════════
#  报告生成
# ══════════════════════════════════════════════════════════════
def generate_synthetic_report(task_a_records, task_b_result, task_c_results):
    """生成 synthetic_report.md (覆盖 Task A + B)"""
    lines = []

    def w(s=""):
        lines.append(s)

    w("# Synthetic High-Dimensional Data Report (Task A & B)")
    w()
    w("## 数据生成机制 (DGP)")
    w()
    w("- **样本量:** n = 150")
    w("- **特征数:** p = 80 (可超训练样本数)")
    w("- **潜在因子结构:** 5 个潜在因子 F0~F4，每个 ~ N(0,1)")
    w("- **观测变量生成:** 每个 X_j 是 2~3 个随机潜在因子的线性组合 + N(0, 0.3²)")
    w("- **目标变量:** y = 2.0 * F0 + 1.5 * F1 + N(0, 0.5²)")
    w()
    w("这是一个典型的 **高维 + 信息冗余** 数据：")
    w("- p 接近甚至超过训练样本量，OLS 面临自由度不足")
    w("- 大量特征共享同一组潜在因子，存在严重多重共线性")
    w("- 真正驱动 y 的只有 2 个潜在方向，信息高度集中在低维子空间中")
    w()

    w("## Task A: OLS 在高维场景下的失败")
    w()
    w("### A3. 不同特征维度下的 OLS 表现")
    w()
    w("| p | rank(X_train) | Condition Number | Train RMSE | Test RMSE |")
    w("|---|--------------|-----------------|------------|-----------|")
    for r in task_a_records:
        w(f"| {r['p']} | {r['rank']} | {r['cond_num']:.2e} | {r['train_rmse']:.4f} | {r['test_rmse']:.4f} |")
    w()
    w("![RMSE and Matrix Structure](task_a_rmse_and_matrix.png)")
    w()
    w("**图说明:**")
    w("- 左图: 横轴=特征数 p, 纵轴=RMSE; 蓝色=训练集, 红色=测试集")
    w("- 右图: 横轴=特征数 p; 柱状图=rank(X_train), 红线=Condition Number (对数刻度)")
    w()
    w("### 为什么训练误差接近 0 反而是危险信号？")
    w()
    w("当 p ≥ n 时，OLS 可以在训练集上完美拟合（或接近完美），")
    w("但这通常意味着模型在\"记忆噪声\"而非学习信号。")
    w("此时 condition number 急剧增大，矩阵近乎奇异，")
    w("系数的方差趋于无穷——换一组样本，系数可能面目全非。")
    w("这就是著名的 **bias-variance tradeoff** 在高维场景下的极端体现。")
    w()

    w("### A4. 系数不稳定性")
    w()
    w("![Coefficient Instability](task_a_coef_instability.png)")
    w()
    w("**图说明:**")
    w("- 横轴=特征名 (选取 X0,X1,X2,X10,X30,X59 共 6 个代表性变量)")
    w("- 纵轴=OLS 回归系数值")
    w("- 每个箱线代表该变量在 50 次不同随机切分下的系数分布")
    w()
    w("**观察:** 同一变量的系数在不同切分下剧烈波动（箱线跨度大），")
    w("说明 OLS 估计极度不稳定。在实际业务中，这意味着：")
    w("> 你无法可靠地告诉业务方\"这个变量到底有多重要\"——")
    w("> 因为换一批数据，答案可能完全相反。")
    w()

    w("## Task B: PCA 与 PCR")
    w()
    w("### B1. PCA 累计解释方差")
    w()
    cum_var = task_b_result["cum_var"]
    k_80 = int(np.argmax(cum_var >= 0.8) + 1)
    k_90 = int(np.argmax(cum_var >= 0.9) + 1)
    k_95 = int(np.argmax(cum_var >= 0.95) + 1)
    w(f"- 80% 方差需要 k={k_80} 个主成分")
    w(f"- 90% 方差需要 k={k_90} 个主成分")
    w(f"- 95% 方差需要 k={k_95} 个主成分")
    w()
    w("![PCA Variance](task_b_pca_variance.png)")
    w()
    w("**图说明:** 横轴=主成分数 k, 纵轴=累计解释方差比例。")
    w("虚线标注了 80%/90%/95% 阈值。")
    w()
    w("**解释:** 原始 80 维空间其实贴近一个更低的 k 维子空间。")
    w("前几个主成分就'吃下'了大部分方差，说明信息高度冗余、")
    w("原始数据存在强共线性——这正好印证了 OLS 不稳定的根源。")
    w()

    w("### B2. PCR 在不同 k 下的表现")
    w()
    bk = task_b_result["best_k"]
    w(f"最优 k (min CV RMSE) = **{bk}**")
    w(f"OLS 基线: train_rmse={task_b_result['ols_train_rmse']:.4f}, "
      f"test_rmse={task_b_result['ols_test_rmse']:.4f}")
    w()
    w("![PCR Curves](task_b_pcr_curves.png)")
    w()
    w("**图说明:**")
    w("- 横轴=保留的主成分数 k")
    w("- 纵轴=RMSE")
    w("- 蓝线=训练集 RMSE, 红线=测试集 RMSE, 绿线=5折 CV RMSE")
    w("- 橙色虚线=OLS 测试集 RMSE 作为基线")
    w()

    w("### B3. CV 曲线解释")
    w()
    w("**PCR CV RMSE** 代表在未见过的验证折上的平均预测误差，是模型泛化能力的无偏估计。")
    w()
    w("- **train 曲线** 单调下降：更多 PC → 更多信息 → 训练误差更低")
    w("- **test/CV 曲线** 呈 U 型：先降后升。")
    w("  太少 PC 会欠拟合（丢失有用信号），太多 PC 会引入噪声方向导致过拟合")
    w("- OLS 在原始高维空间训练误差极低甚至为 0，但测试误差很高——")
    w("  因为它把噪声方向也当作信号来拟合了。PCR 通过舍弃小方差方向避免了这一问题")
    w()

    w("### B4. 公式与定义")
    w()
    w("**OLS 估计式:**")
    w()
    w("$$\\hat{\\beta}_{OLS} = (X^T X)^{-1} X^T y$$")
    w()
    w("**第一主成分的方差最大化定义:**")
    w()
    w("$$v_1 = \\arg\\max_{\\|v\\|=1} \\text{Var}(X v) = \\arg\\max_{\\|v\\|=1} v^T \\Sigma v$$")
    w()
    w("其中 $\\Sigma = \\frac{1}{n} X^T X$ 是协方差矩阵（已中心化）。")
    w()
    w("**PCR 流程:**")
    w()
    w("1. 标准化: $\\tilde{X} = (X - \\mu) / \\sigma$")
    w("2. PCA: $\\tilde{X} = U S V^T$, 取 $V_k$ (前 k 列)")
    w("3. 投影: $Z_k = \\tilde{X} V_k$")
    w("4. 回归: $\\hat{y} = Z_k \\hat{\\gamma}_k$, 其中 $\\hat{\\gamma}_k = (Z_k^T Z_k)^{-1} Z_k^T y$")
    w("5. 原始尺度系数: $\\hat{\\beta}_{PCR} = V_k \\hat{\\gamma}_k / \\sigma$")
    w()

    SYNTHETIC_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {SYNTHETIC_MD} 已生成")


def generate_summary_comparison(task_c_results):
    """生成 summary_comparison.md (Task C)"""
    lines = []

    def w(s=""):
        lines.append(s)

    w("# Lasso vs PCR: Selection vs Compression (Task C)")
    w()
    w("## C1 & C2. 两种数据世界中的对比")
    w()

    for world_name, res in task_c_results.items():
        w(f"### {world_name}")
        w()
        w("| 方法 | Test RMSE | Test MAE | 复杂度指标 | 稳定性 |")
        w("|------|----------|----------|-----------|--------|")
        for method in ["OLS", "Lasso", "PCR"]:
            r = res[method]
            if method == "Lasso":
                complexity = f"非零系数={r['n_nonzero']}"
                stab = f"{r.get('stability', 'N/A'):.4f}" if isinstance(r.get('stability'), float) else "N/A"
            elif method == "PCR":
                complexity = f"k={r['k']}"
                stab = f"{r.get('stability', 'N/A'):.4f}" if isinstance(r.get('stability'), float) else "N/A"
            else:
                complexity = "全部变量"
                stab = "—"
            w(f"| {method} | {r['rmse']:.4f} | {r['mae']:.4f} | {complexity} | {stab} |")
        w()

    w("![Lasso vs PCR](task_c_lasso_vs_pcr.png)")
    w()
    w("**图说明:** 左右两图分别对应 Sparse Truth 和 Latent-Factor Truth 场景。")
    w("横轴=方法 (OLS/Lasso/PCR), 纵轴=测试集 RMSE, 柱上数字为具体数值。")
    w()

    w("## C3. 核心问题讨论")
    w()
    w("### Q1: 当数据真的是 sparse truth 时，为什么 Lasso 往往更自然？")
    w()
    w("Sparse truth 的本质是\"只有少数原始变量真正重要\"。")
    w("Lasso 的 L1 惩罚天然倾向于产生稀疏解——将不重要变量的系数压缩为零，")
    w("直接给出一个\"谁留下、谁走开\"的清晰答案。这恰好匹配 sparse truth 的生成机制。")
    w()
    w("### Q2: 当数据更像 latent-factor truth 时，为什么 PCR 往往更自然？")
    w()
    w("Latent-factor truth 的本质是\"原始变量只是潜在因子的投影\"。")
    w("此时没有一个原始变量是\"真正重要\"的——重要的是那些方向（主成分）。")
    w("PCR 先找出方差最大的方向，再在这些方向上回归，恰好在做\"信息压缩\"而非\"变量挑选\"。")
    w("它不关心某个具体变量是否留下，而是关心多少信息被保留。")
    w()
    w("### Q3: Lasso 回答的更像\"谁留下\"，PCR 回答的更像什么？")
    w()
    w("Lasso 回答的是: **\"哪些原始变量对 y 有直接的、独立的贡献？\"**")
    w("PCR 回答的是: **\"多少信息维度足以描述 X→y 的关系？\"**")
    w()
    w("一个做 **selection**（挑选），一个做 **compression**（压缩）。")
    w()
    w("### Q4: 如果业务方要求\"一个更短的变量名单\"，你更可能用哪个方法？")
    w()
    w("**Lasso。** 因为它直接在原始变量空间中给出非零系数名单，")
    w("业务方可以直接理解\"这 5 个变量是关键的\"。")
    w("而 PCR 的主成分是原始变量的线性组合，难以用业务语言解释。")
    w()
    w("### Q5: 如果业务方要求\"一个更稳的预测器\"，你更可能用哪个方法？")
    w()
    w("**PCR。** 因为 PCR 舍弃了噪声方向（小方差主成分），只保留信息最集中的方向，")
    w("在高维共线性场景下通常比 Lasso 更稳定——它不受\"同一族中选哪个\"的困扰。")
    w("但最终选择也取决于数据更接近哪种生成机制。")
    w()

    w("## C4. 为什么不把前向/后向选择拉回主舞台？")
    w()
    w("本周主线是 **selection vs compression** 这一对更高层次的方法论对比。")
    w("前向/后向选择本质上也属于 selection 路线（在原始变量中挑子集），")
    w("与 Lasso 属于同一阵营——因此把它拉回来会淡化\"压缩\"这条线。")
    w()
    w("如果要归类：前向/后向选择更接近 **selection** 路线——")
    w("它回答的也是\"哪些列有用\"，而不是\"数据本身有多高维\"。")
    w()

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {SUMMARY_MD} 已生成")


def generate_kaggle_report(task_d_result):
    """生成 kaggle_report.md (Task D)"""
    if task_d_result is None:
        return

    lines = []

    def w(s=""):
        lines.append(s)

    w("# Real Data Report: Housing (Task D)")
    w()
    w("## 数据来源与背景")
    w("- **数据集:** Housing (housing.csv)")
    w(f"- **样本数:** {task_d_result.get('n_samples', 'N/A')}")
    w(f"- **特征数:** {task_d_result.get('n_features', 'N/A')}")
    w()

    r = task_d_result
    w("## 诊断信息")
    w()
    w(f"- rank(X_train) = {r['rank']}")
    w(f"- Condition Number = {r['cond_num']:.2e}")
    if 'cum_var_5' in r and not np.isnan(r['cum_var_5']):
        w(f"- 前 5 个 PC 解释方差比例 = {r['cum_var_5']*100:.1f}%")
    w()

    w("## 测试集表现")
    w()
    w("| 方法 | RMSE | MAE | 复杂度 |")
    w("|------|------|-----|--------|")
    w(f"| OLS   | {r['OLS']['rmse']:.4f} | {r['OLS']['mae']:.4f} | "
      f"{r.get('n_features_used', '全部变量')} |")
    w(f"| Lasso | {r['Lasso']['rmse']:.4f} | {r['Lasso']['mae']:.4f} | "
      f"非零={r['Lasso']['n_nonzero']}/{r.get('n_features_used', '?')} |")
    w(f"| PCR   | {r['PCR']['rmse']:.4f} | {r['PCR']['mae']:.4f} | "
      f"k={r['PCR']['k']} |")
    w()

    w("## Lasso 系数 (Top-15 by |coef|)")
    w()
    fc = r.get("feature_cols", [])
    lc = r["Lasso"]["coef"]
    if len(fc) == len(lc):
        sorted_coefs = sorted(zip(fc, lc), key=lambda x: -abs(x[1]))
        w("| 特征 | 系数 |")
        w("|------|------|")
        for fn, cv in sorted_coefs[:15]:
            marker = " ← 为零" if abs(cv) < 1e-8 else ""
            w(f"| {fn} | {cv:.6f}{marker} |")
    w()

    w("## 真实数据解释")
    w()

    w("### OLS 是否出现了高维/共线性不稳定迹象？")
    w()
    if r["cond_num"] > 100:
        w(f"Condition Number = {r['cond_num']:.1f}，存在中等及以上共线性。")
    else:
        w(f"Condition Number = {r['cond_num']:.1f}，共线性尚可接受。")
    w()

    w("### Lasso 与 PCR 谁表现更好？为什么？")
    w()
    lasso_rmse = r["Lasso"]["rmse"]
    pcr_rmse  = r["PCR"]["rmse"]
    if lasso_rmse < pcr_rmse:
        w(f"Lasso (RMSE={lasso_rmse:.4f}) 优于 PCR (RMSE={pcr_rmse:.4f})。")
        w("数据可能更接近 sparse truth。")
    elif pcr_rmse < lasso_rmse:
        w(f"PCR (RMSE={pcr_rmse:.4f}) 优于 Lasso (RMSE={lasso_rmse:.4f})。")
        w("数据可能更接近 latent-factor truth。")
    else:
        w(f"两者表现接近 (Lasso={lasso_rmse:.4f}, PCR={pcr_rmse:.4f})。")
    w()

    w("### 这份数据到底适合筛选还是压缩？")
    w()
    if 'cum_var_5' in r and not np.isnan(r['cum_var_5']):
        cum5 = r["cum_var_5"]
        if cum5 > 0.85:
            w(f"前 5 个 PC 解释了 {cum5*100:.1f}% 方差 → 偏 **压缩 (PCR)**。")
        else:
            w(f"前 5 个 PC 仅解释 {cum5*100:.1f}% 方差 → 信息分散，偏 **筛选 (Lasso)**。")
    w("最终选择取决于业务需求：可解释的变量名单 → Lasso；稳定预测 → PCR。")
    w()

    KAGGLE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {KAGGLE_MD} 已生成")


# ══════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("Week 14: High-Dimensional Regression, PCA, and PCR")
    print("=" * 70)

    # Task A
    task_a_records, _ = task_a()

    # Task B
    task_b_result = task_b()

    # Task C
    task_c_results = task_c()

    # Task D (optional)
    task_d_result = task_d()

    # ── 生成报告 ──
    print("\n" + "=" * 70)
    print("生成报告中...")
    print("=" * 70)

    generate_synthetic_report(task_a_records, task_b_result, task_c_results)
    generate_summary_comparison(task_c_results)
    generate_kaggle_report(task_d_result)

    print("\n" + "=" * 70)
    print("All tasks completed. 查看 results/ 目录下的报告。")
    print("=" * 70)
