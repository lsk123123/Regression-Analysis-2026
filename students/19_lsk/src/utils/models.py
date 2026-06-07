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
# 变量选择：前向选择 & 后向剔除（基于交叉验证）
# ============================================================

def forward_selection(
    X: np.ndarray,
    y: np.ndarray,
    max_features: int = None,
    cv_folds: int = 5,
    scoring: str = "rmse",
    verbose: bool = True,
) -> tuple:
    """
    基于交叉验证的前向选择（Forward Selection）。

    从空模型开始，每次添加一个对 CV 性能提升最大的特征，
    直到达到 max_features 或无法再提升。

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        特征矩阵（不含截距列，函数内部会自动添加）
    y : np.ndarray, shape (n,)
        目标变量
    max_features : int or None
        最大选择特征数，None 表示最多选全部特征
    cv_folds : int
        交叉验证折数
    scoring : str
        评分方式: "rmse" 或 "r2"
    verbose : bool
        是否打印中间过程

    Returns
    -------
    selected_indices : list
        被选中特征的列索引
    cv_scores_history : list
        每一步的 CV 分数
    """
    n_samples, n_features = X.shape
    if max_features is None:
        max_features = n_features

    remaining = list(range(n_features))
    selected = []
    best_cv_score = np.inf if scoring == "rmse" else -np.inf
    cv_scores_history = []

    # 手动 K-Fold 划分
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // cv_folds

    if verbose:
        print(f"\n{'='*60}")
        print(f"前向选择 (Forward Selection) - max_features={max_features}")
        print(f"{'='*60}")

    for step in range(max_features):
        best_feature = None
        best_feature_score = np.inf if scoring == "rmse" else -np.inf

        for feat in remaining:
            candidate_features = selected + [feat]
            X_candidate = X[:, candidate_features]

            # 交叉验证
            cv_scores = []
            for fold in range(cv_folds):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_train, y_train = X_candidate[train_idx], y[train_idx]
                X_val, y_val = X_candidate[val_idx], y[val_idx]

                # 添加截距列
                X_train_c = np.column_stack([np.ones(X_train.shape[0]), X_train])
                X_val_c = np.column_stack([np.ones(X_val.shape[0]), X_val])

                try:
                    model = AnalyticalOLS()
                    model.fit(X_train_c, y_train)
                    y_pred = model.predict(X_val_c)
                    if scoring == "rmse":
                        score = np.sqrt(np.mean((y_val - y_pred) ** 2))
                    else:
                        sse = np.sum((y_val - y_pred) ** 2)
                        sst = np.sum((y_val - np.mean(y_val)) ** 2)
                        score = 1 - sse / sst if sst != 0 else 0.0
                    cv_scores.append(score)
                except np.linalg.LinAlgError:
                    cv_scores.append(np.inf if scoring == "rmse" else -np.inf)

            avg_score = np.mean(cv_scores)

            if scoring == "rmse":
                if avg_score < best_feature_score:
                    best_feature_score = avg_score
                    best_feature = feat
            else:
                if avg_score > best_feature_score:
                    best_feature_score = avg_score
                    best_feature = feat

        # 检查是否有提升
        improved = False
        if scoring == "rmse":
            if best_feature_score < best_cv_score:
                improved = True
        else:
            if best_feature_score > best_cv_score:
                improved = True

        if best_feature is not None and improved:
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_cv_score = best_feature_score
            cv_scores_history.append(best_cv_score)
            if verbose:
                print(f"  Step {step+1}: 添加特征 {best_feature}, CV {scoring.upper()}={best_feature_score:.6f}")
        else:
            if verbose:
                print(f"  Step {step+1}: 无法再提升，停止选择")
            break

    if verbose:
        print(f"  最终选中特征: {selected}")
        print(f"{'='*60}\n")

    return selected, cv_scores_history


def backward_elimination(
    X: np.ndarray,
    y: np.ndarray,
    min_features: int = 1,
    cv_folds: int = 5,
    scoring: str = "rmse",
    verbose: bool = True,
) -> tuple:
    """
    基于交叉验证的后向剔除（Backward Elimination）。

    从全模型开始，每次删除一个对 CV 性能影响最小（或删除后性能最好）的特征，
    直到达到 min_features。

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        特征矩阵（不含截距列，函数内部会自动添加）
    y : np.ndarray, shape (n,)
        目标变量
    min_features : int
        最少保留的特征数
    cv_folds : int
        交叉验证折数
    scoring : str
        评分方式: "rmse" 或 "r2"
    verbose : bool
        是否打印中间过程

    Returns
    -------
    selected_indices : list
        被保留特征的列索引
    cv_scores_history : list
        每一步的 CV 分数
    """
    n_samples, n_features = X.shape

    selected = list(range(n_features))
    cv_scores_history = []

    # 手动 K-Fold 划分
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // cv_folds

    # 评估全模型
    X_full = X[:, selected]
    full_cv_scores = []
    for fold in range(cv_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        X_train_c = np.column_stack([np.ones(len(train_idx)), X_full[train_idx]])
        X_val_c = np.column_stack([np.ones(len(val_idx)), X_full[val_idx]])
        try:
            model = AnalyticalOLS()
            model.fit(X_train_c, y[train_idx])
            y_pred = model.predict(X_val_c)
            if scoring == "rmse":
                full_cv_scores.append(np.sqrt(np.mean((y[val_idx] - y_pred) ** 2)))
            else:
                sse = np.sum((y[val_idx] - y_pred) ** 2)
                sst = np.sum((y[val_idx] - np.mean(y[val_idx])) ** 2)
                full_cv_scores.append(1 - sse / sst if sst != 0 else 0.0)
        except np.linalg.LinAlgError:
            full_cv_scores.append(np.inf if scoring == "rmse" else -np.inf)

    current_best = np.mean(full_cv_scores)
    cv_scores_history.append(current_best)

    if verbose:
        print(f"\n{'='*60}")
        print(f"后向剔除 (Backward Elimination) - min_features={min_features}")
        print(f"{'='*60}")
        print(f"  Step 0 (全模型): {len(selected)} 特征, CV {scoring.upper()}={current_best:.6f}")

    for step in range(n_features - min_features):
        worst_feature = None
        best_after_removal = np.inf if scoring == "rmse" else -np.inf

        for feat in selected:
            candidate_features = [f for f in selected if f != feat]
            if len(candidate_features) == 0:
                continue
            X_candidate = X[:, candidate_features]

            cv_scores = []
            for fold in range(cv_folds):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                X_train_c = np.column_stack([np.ones(len(train_idx)), X_candidate[train_idx]])
                X_val_c = np.column_stack([np.ones(len(val_idx)), X_candidate[val_idx]])
                try:
                    model = AnalyticalOLS()
                    model.fit(X_train_c, y[train_idx])
                    y_pred = model.predict(X_val_c)
                    if scoring == "rmse":
                        cv_scores.append(np.sqrt(np.mean((y[val_idx] - y_pred) ** 2)))
                    else:
                        sse = np.sum((y[val_idx] - y_pred) ** 2)
                        sst = np.sum((y[val_idx] - np.mean(y[val_idx])) ** 2)
                        cv_scores.append(1 - sse / sst if sst != 0 else 0.0)
                except np.linalg.LinAlgError:
                    cv_scores.append(np.inf if scoring == "rmse" else -np.inf)

            avg_score = np.mean(cv_scores)

            if scoring == "rmse":
                if avg_score < best_after_removal:
                    best_after_removal = avg_score
                    worst_feature = feat
            else:
                if avg_score > best_after_removal:
                    best_after_removal = avg_score
                    worst_feature = feat

        if worst_feature is not None:
            selected.remove(worst_feature)
            current_best = best_after_removal
            cv_scores_history.append(current_best)
            if verbose:
                print(f"  Step {step+1}: 剔除特征 {worst_feature}, CV {scoring.upper()}={current_best:.6f}")
        else:
            break

    if verbose:
        print(f"  最终保留特征: {selected}")
        print(f"{'='*60}\n")

    return selected, cv_scores_history