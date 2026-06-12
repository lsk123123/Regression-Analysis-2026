"""
模块：utils.models
核心机器学习模型：解析解OLS（正规方程）和梯度下降OLS。
"""
import numpy as np
from scipy import stats


class AnalyticalOLS:
    """
    解析解线性回归（正规方程）。
    假设 X 已经包含截距列（全1），或由外部添加。
    """
    def __init__(self):
        self.coef_ = None          # 回归系数
        self.cov_matrix_ = None    # 协方差矩阵
        self.sigma2_ = None        # 残差方差
        self.df_resid_ = None      # 残差自由度
        self.resid_ = None          # 残差
        self._feature_names = None  # 特征名称（可选）

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        """
        拟合模型：β = (XᵀX)⁻¹ Xᵀy
        """
        n, p = X.shape
        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = [f'X{i}' for i in range(p)]

        # 正规方程
        XtX = X.T @ X
        Xty = X.T @ y
        self.coef_ = np.linalg.solve(XtX, Xty)

        # 残差与自由度
        y_pred = X @ self.coef_
        self.resid_ = y - y_pred
        self.df_resid_ = n - p
        sse = np.sum(self.resid_ ** 2)
        self.sigma2_ = sse / self.df_resid_

        # 协方差矩阵
        XtX_inv = np.linalg.inv(XtX)
        self.cov_matrix_ = self.sigma2_ * XtX_inv
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 y = Xβ"""
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """决定系数 R² = 1 - SSE/SST"""
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst if sst != 0 else 0.0

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """一般线性假设检验 Cβ = d，返回 F 统计量和 p 值"""
        if self.coef_ is None:
            raise ValueError("模型尚未拟合")
        C = np.atleast_2d(C)
        q = C.shape[0]
        diff = C @ self.coef_ - d
        C_cov_Ct = C @ self.cov_matrix_ @ C.T
        try:
            inv = np.linalg.inv(C_cov_Ct)
            f_stat = (diff.T @ inv @ diff) / q
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(C_cov_Ct)
            f_stat = (diff.T @ inv @ diff) / q
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        return {'f_stat': f_stat, 'p_value': p_value, 'df_num': q, 'df_den': self.df_resid_}

    def summary(self) -> str:
        """打印模型摘要"""
        if self.coef_ is None:
            return "模型尚未拟合"
        std_errors = np.sqrt(np.diag(self.cov_matrix_))
        t_stats = self.coef_ / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.df_resid_))
        lines = []
        lines.append("=" * 70)
        lines.append("                     AnalyticalOLS 回归结果")
        lines.append("=" * 70)
        lines.append(f"残差自由度: {self.df_resid_}")
        lines.append(f"σ̂² (残差方差): {self.sigma2_:.6f}\n")
        lines.append(f"{'变量':<15} {'系数':>12} {'标准误':>12} {'t统计量':>12} {'p值':>12}")
        lines.append("-" * 70)
        for i, (coef, se, t, p) in enumerate(zip(self.coef_, std_errors, t_stats, p_values)):
            name = self._feature_names[i] if i < len(self._feature_names) else f"X{i}"
            lines.append(f"{name:<15} {coef:>12.6f} {se:>12.6f} {t:>12.6f} {p:>12.4e}")
        lines.append("=" * 70)
        return "\n".join(lines)


class GradientDescentOLS:
    """
    梯度下降线性回归。
    支持全批量 (full_batch) 和小批量 (mini_batch)。
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        """梯度下降拟合模型，X 应已包含截距列"""
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)

        # 确定批量大小
        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type 必须是 'full_batch' 或 'mini_batch'")

        for epoch in range(self.max_iter):
            # 小批量采样
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            # 计算梯度 (MSE)
            y_pred_batch = X_batch @ self.coef_
            error = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error)

            # 更新系数
            self.coef_ -= self.learning_rate * gradient

            # 记录全量损失（用于收敛判断）
            y_pred_full = X @ self.coef_
            mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(mse)

            # 收敛检查
            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst if sst != 0 else 0.0


# ============================================================
# PCR: Principal Component Regression
# ============================================================
class PCR:
    """
    主成分回归 (Principal Component Regression)。

    工作流:
        1. 标准化 X
        2. 对标准化后的 X 做 PCA
        3. 取前 k 个主成分得分 Z_k = X_scaled @ V_k
        4. 在 Z_k 上用 OLS 回归 y

    Parameters
    ----------
    n_components : int
        保留的主成分个数 k
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.mean_ = None
        self.std_ = None
        self.components_ = None          # V_k: (p, k)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.coef_pc_ = None             # 在主成分空间中的系数
        self.coef_original_ = None       # 映射回原始变量空间的系数
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合 PCR 模型。

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            原始特征矩阵（不含截距）
        y : np.ndarray, shape (n,)
            目标变量
        """
        n, p = X.shape
        k = min(self.n_components, min(n, p))

        # Step 1: 标准化
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        X_scaled = (X - self.mean_) / self.std_

        # Step 2: PCA via SVD
        # X_scaled = U @ S @ Vt
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        self.explained_variance_ = (S ** 2) / n
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        self.components_ = Vt[:k, :].T    # (p, k)

        # Step 3: 主成分得分 Z_k = X_scaled @ V_k
        Z_k = X_scaled @ self.components_  # (n, k)
        Z_k_with_intercept = np.column_stack([np.ones(n), Z_k])

        # Step 4: OLS on Z_k
        ols = AnalyticalOLS()
        ols.fit(Z_k_with_intercept, y)
        self.intercept_ = ols.coef_[0]
        self.coef_pc_ = ols.coef_[1:]     # 在 PC 空间中的系数 (k,)

        # 映射回原始变量空间: β_original = V_k @ coef_pc_ / std
        self.coef_original_ = (self.components_ @ self.coef_pc_) / self.std_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X_scaled = (X - self.mean_) / self.std_
        Z = X_scaled @ self.components_
        return self.intercept_ + Z @ self.coef_pc_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R²"""
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst if sst != 0 else 0.0


# ============================================================
# 稳定性指标: 系数标准差
# ============================================================
def coefficient_stability(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 50,
    test_size: float = 0.3,
    model_factory=None,
) -> dict:
    """
    测量回归系数在不同随机切分下的稳定性。

    返回值中的 "std" 是各系数跨切分的标准差，越小越稳定。

    Parameters
    ----------
    X, y : 数据
    n_splits : 切分次数
    test_size : 测试集比例（仅用于切分，稳定性只看训练集上的系数）
    model_factory : callable, 返回一个具有 .fit(X_train, y_train) 和 .coef_ 的模型

    Returns
    -------
    dict with keys: 'coefs_matrix', 'coefs_mean', 'coefs_std', 'stability_score'
    """
    if model_factory is None:
        model_factory = lambda: AnalyticalOLS()

    coefs = []
    for seed in range(n_splits):
        X_tr, _, y_tr, _ = train_test_split_wrapper(X, y, test_size=test_size, seed=seed)
        model = model_factory()
        # 为 OLS 添加截距
        X_tr_c = np.column_stack([np.ones(X_tr.shape[0]), X_tr])
        model.fit(X_tr_c, y_tr)
        coefs.append(model.coef_[1:].copy())   # 去掉截距

    coefs = np.array(coefs)
    return {
        "coefs_matrix": coefs,
        "coefs_mean": np.mean(coefs, axis=0),
        "coefs_std": np.std(coefs, axis=0),
        "stability_score": np.mean(np.std(coefs, axis=0)),
    }


# ============================================================
# 数据生成器
# ============================================================
def make_high_dimensional_data(
    n_samples: int = 120,
    n_features: int = 60,
    n_latent_factors: int = 5,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple:
    """
    生成具有潜在低秩因子的高维回归数据。

    结构:
        - n_latent_factors 个潜在因子 F (每组 N(0,1))
        - 每个观测特征 X_j 是 2~3 个随机因子的线性组合 + 噪声
        - y = 2*F0 + 1.5*F1 + noise (仅前 2 个因子驱动 y)

    Parameters
    ----------
    n_samples : int, 样本数
    n_features : int, 特征数 (p)
    n_latent_factors : int, 潜在因子数
    noise_std : float, 噪声标准差
    seed : int, 随机种子

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
    y : np.ndarray (n_samples,)
    F : np.ndarray (n_samples, n_latent_factors)  潜在因子
    """
    rng = np.random.default_rng(seed)

    # 潜在因子
    F = rng.normal(0, 1, (n_samples, n_latent_factors))

    # 每个特征由随机选中的 2~3 个因子线性组合 + 噪声
    X = np.zeros((n_samples, n_features))
    for j in range(n_features):
        # 随机选 2~3 个因子
        n_factors_for_j = rng.integers(2, 4)
        factor_indices = rng.choice(n_latent_factors, size=n_factors_for_j, replace=False)
        weights = rng.uniform(0.3, 1.5, size=n_factors_for_j)
        X[:, j] = F[:, factor_indices] @ weights + rng.normal(0, 0.3, n_samples)

    # y 仅由前 2 个因子驱动
    eps = rng.normal(0, noise_std, n_samples)
    y = 2.0 * F[:, 0] + 1.5 * F[:, 1] + eps

    return X, y, F


def make_sparse_regression_data(
    n_samples: int = 120,
    n_features: int = 60,
    n_true_features: int = 5,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple:
    """
    生成稀疏真实模型数据：只有少数原始变量直接决定 y。

    结构:
        - 前 n_true_features 个特征是真实信号 (N(0,1))
        - 其余是纯噪声 N(0,1)
        - y = 2*X0 + 1.5*X1 + 0.8*X2 - 1.0*X3 + 0.5*X4 + noise

    Parameters
    ----------
    n_samples : int, 样本数
    n_features : int, 特征数 (p)
    n_true_features : int, 真实信号特征数
    noise_std : float, 噪声标准差
    seed : int, 随机种子

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
    y : np.ndarray (n_samples,)
    true_coef : np.ndarray (n_features,)   真实系数
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))

    true_coef = np.zeros(n_features)
    # 设置真实信号系数（只在前 n_true_features 个特征上）
    signal_coefs = np.array([2.0, 1.5, 0.8, -1.0, 0.5])
    true_coef[:n_true_features] = signal_coefs[:n_true_features]

    eps = rng.normal(0, noise_std, n_samples)
    y = X @ true_coef + eps

    return X, y, true_coef


# ============================================================
# 辅助: train_test_split (简单封装)
# ============================================================
def train_test_split_wrapper(X, y, test_size=0.3, seed=42):
    """简单的确定性切分"""
    rng = np.random.default_rng(seed)
    n = len(y)
    n_test = int(n * test_size)
    indices = rng.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]