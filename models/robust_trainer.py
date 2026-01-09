"""
训练稳健性优化模块 (Training Robustness)

核心功能:
1. Purged Cross-Validation - 防止特征重叠导致的信息泄露
2. Sample Weighting - 时间衰减 + 回报加权
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Tuple
from dataclasses import dataclass
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RobustTrainConfig:
    """稳健训练配置"""
    # Purged CV 配置
    n_splits: int = 5             # 交叉验证折数
    purge_periods: int = 5        # 清除的周期数 (防止前瞻偏差)
    embargo_periods: int = 3      # 禁止使用的周期数
    
    # 样本加权配置
    use_time_decay: bool = True   # 是否启用时间衰减
    decay_half_life_days: int = 90  # 半衰期 (天)
    
    use_return_weight: bool = True  # 是否启用回报加权
    extreme_quantile: float = 0.9   # 极端回报分位数
    extreme_boost: float = 1.5      # 极端样本权重倍数


class RobustTrainer:
    """
    稳健模型训练器
    
    自动集成:
    - Purged Cross-Validation (防止信息泄露)
    - Sample Weighting (时间衰减 + 回报加权)
    """
    
    def __init__(self, config: RobustTrainConfig = None):
        self.config = config or RobustTrainConfig()
        self.model = None
        self.cv_scores = []
    
    def compute_sample_weights(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        timestamp_col: str = 'timestamp'
    ) -> np.ndarray:
        """计算样本权重 (时间衰减 + 回报加权)"""
        n = len(df)
        weights = np.ones(n)
        
        # 1. 时间衰减权重: 越近期的样本权重越高
        if self.config.use_time_decay and timestamp_col in df.columns:
            timestamps = pd.to_datetime(df[timestamp_col])
            most_recent = timestamps.max()
            days_ago = (most_recent - timestamps).dt.total_seconds() / 86400
            
            decay_lambda = np.log(2) / self.config.decay_half_life_days
            weights *= np.exp(-decay_lambda * days_ago)
        
        # 2. 回报加权: 极端样本获得更高权重
        if self.config.use_return_weight and target_col in df.columns:
            abs_target = np.abs(df[target_col])
            threshold = np.percentile(abs_target, self.config.extreme_quantile * 100)
            extreme_mask = abs_target > threshold
            weights[extreme_mask] *= self.config.extreme_boost
        
        # 归一化
        weights = weights * (n / weights.sum())
        return weights
    
    def purged_cv_split(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str = 'timestamp'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成 Purged K-Fold 分割 (防止信息泄露)"""
        df = df.copy()
        df['_index'] = np.arange(len(df))
        
        unique_times = df[timestamp_col].sort_values().unique()
        n_times = len(unique_times)
        fold_size = n_times // self.config.n_splits
        
        splits = []
        
        for i in range(self.config.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.config.n_splits - 1 else n_times
            test_times = unique_times[test_start:test_end]
            
            # Purge: 移除与测试集相邻的训练样本
            purge_start = max(0, test_start - self.config.purge_periods)
            embargo_end = min(n_times, test_end + self.config.embargo_periods)
            
            train_times = np.concatenate([
                unique_times[:purge_start],
                unique_times[embargo_end:]
            ])
            
            train_mask = df[timestamp_col].isin(train_times)
            test_mask = df[timestamp_col].isin(test_times)
            
            train_idx = df.loc[train_mask, '_index'].values
            test_idx = df.loc[test_mask, '_index'].values
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits
    
    def train_ranker(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        timestamp_col: str = 'timestamp',
        model_params: dict = None
    ) -> Tuple[lgb.LGBMRanker, dict]:
        """
        训练排序模型 (LGBMRanker)
        
        Returns:
            (model, results) 元组
        """
        print("🛡️ 稳健训练: LGBMRanker (Purged CV + 样本加权)")
        
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        weights = self.compute_sample_weights(df, target_col, timestamp_col)
        splits = self.purged_cv_split(df, timestamp_col)
        
        params = model_params or {
            "objective": "lambdarank", "metric": "ndcg",
            "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 100,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "random_state": 42, "verbosity": -1
        }
        
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = df.iloc[train_idx][feature_cols], df.iloc[train_idx][target_col]
            X_test, y_test = df.iloc[test_idx][feature_cols], df.iloc[test_idx][target_col]
            
            train_groups = df.iloc[train_idx].groupby(timestamp_col).size().tolist()
            test_groups = df.iloc[test_idx].groupby(timestamp_col).size().tolist()
            
            model = lgb.LGBMRanker(
                **params,
                label_gain=np.arange(max(y_train.max(), y_test.max()) + 1).tolist()
            )
            
            model.fit(
                X_train, y_train,
                group=train_groups,
                sample_weight=weights[train_idx],
                eval_set=[(X_test, y_test)],
                eval_group=[test_groups],
                eval_at=[1, 3],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            
            score = model.best_score_['valid_0'].get('ndcg@3', 0)
            cv_scores.append(score)
            print(f"  Fold {fold + 1}: NDCG@3 = {score:.4f}")
        
        # 最终模型: 全量数据训练
        print("\n📊 在全量数据上训练最终模型...")
        groups = df.groupby(timestamp_col).size().tolist()
        
        self.model = lgb.LGBMRanker(
            **params,
            label_gain=np.arange(int(df[target_col].max()) + 1).tolist()
        )
        self.model.fit(df[feature_cols], df[target_col], group=groups, sample_weight=weights)
        
        results = {
            'cv_scores': cv_scores,
            'mean_ndcg': np.mean(cv_scores),
            'std_ndcg': np.std(cv_scores)
        }
        
        print(f"\n✅ 完成! 平均 NDCG@3: {results['mean_ndcg']:.4f} ± {results['std_ndcg']:.4f}")
        return self.model, results
    
    def train_classifier(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        timestamp_col: str = 'timestamp',
        model_params: dict = None
    ) -> Tuple[lgb.LGBMClassifier, dict]:
        """
        训练分类模型 (LGBMClassifier)
        
        Returns:
            (model, results) 元组
        """
        print("🛡️ 稳健训练: LGBMClassifier (Purged CV + 样本加权)")
        
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        weights = self.compute_sample_weights(df, target_col, timestamp_col)
        splits = self.purged_cv_split(df, timestamp_col)
        
        params = model_params or {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200,
            "random_state": 42, "verbosity": -1
        }
        
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = df.iloc[train_idx][feature_cols], df.iloc[train_idx][target_col]
            X_test, y_test = df.iloc[test_idx][feature_cols], df.iloc[test_idx][target_col]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                sample_weight=weights[train_idx],
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average='macro')
            cv_scores.append(score)
            print(f"  Fold {fold + 1}: F1-Macro = {score:.4f}")
        
        # 最终模型
        print("\n📊 在全量数据上训练最终模型...")
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(df[feature_cols], df[target_col], sample_weight=weights)
        
        results = {
            'cv_scores': cv_scores,
            'mean_f1': np.mean(cv_scores),
            'std_f1': np.std(cv_scores)
        }
        
        print(f"\n✅ 完成! 平均 F1-Macro: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
        return self.model, results


if __name__ == "__main__":
    # 简单测试
    print("=" * 50)
    print("Robust Trainer 模块测试")
    print("=" * 50)
    
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    symbols = ['A', 'B', 'C', 'D', 'E']
    
    data = []
    for d in dates:
        for s in symbols:
            data.append({
                'timestamp': d,
                'symbol': s,
                'f1': np.random.randn(),
                'f2': np.random.randn(),
                'target': np.random.randint(0, 4)
            })
    
    df = pd.DataFrame(data)
    
    trainer = RobustTrainer()
    model, results = trainer.train_ranker(df, ['f1', 'f2'], 'target')
    print(f"\n结果: {results}")
