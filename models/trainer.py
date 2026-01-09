import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import optuna
import json
from pathlib import Path

# 导入稳健训练模块
from models.robust_trainer import RobustTrainer, RobustTrainConfig


class RankingModelTrainer:
    def __init__(self, model_name: str = "Mag7_Ranker"):
        self.model_name = model_name
        self.model = None

    def train(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        target_col: str,
        purge_periods: int = 5,
        use_time_decay: bool = True,
        decay_half_life_days: int = 90
    ) -> dict:
        """
        使用稳健训练方法 (Purged CV + 样本加权)
        
        Args:
            df: 训练数据
            feature_cols: 特征列
            target_col: 目标列
            purge_periods: 样本清除周期数 (防止信息泄露)
            use_time_decay: 是否使用时间衰减权重
            decay_half_life_days: 时间衰减半衰期 (天)
            
        Returns:
            训练结果字典 (包含 cv_scores, mean_ndcg, std_ndcg)
        """
        config = RobustTrainConfig(
            purge_periods=purge_periods,
            use_time_decay=use_time_decay,
            decay_half_life_days=decay_half_life_days
        )
        
        robust_trainer = RobustTrainer(config)
        
        # 加载优化后的超参数
        best_params = self._load_best_params()
        model_params = None
        if best_params:
            print(f"✨ 使用优化后的参数: {best_params}")
            model_params = best_params
        
        self.model, results = robust_trainer.train_ranker(
            df, feature_cols, target_col, model_params=model_params
        )
        
        return results
    
    # 保留别名以兼容旧代码
    train_robust = train


    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, path)
            print(f"Model saved to {path}")
        else:
            print("错误: 没有可保存的模型。")

    def load(self, path: str):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model:
            return self.model.predict(X)
        return np.array([])
    
    def _load_best_params(self) -> dict:
        """从 JSON 文件加载最佳参数"""
        try:
            path = Path('models/best_params.json')
            if path.exists():
                with open(path, 'r') as f:
                    all_params = json.load(f)
                    # 查找对应的模型参数 (这里假设我们只关心 l2_ranker)
                    return all_params.get('l2_ranker', {})
        except Exception as e:
            print(f"⚠️  加载最佳参数失败: {e}")
        return {}

    def optimize(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, n_trials: int = 50) -> dict:
        """使用 Optuna 优化超参数"""
        print(f"\n🔍 开始 Optuna 超参数优化 (L2 Ranker)... 试验次数: {n_trials}")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._objective(trial, df, feature_cols, target_col),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        print(f"\n✅ 优化完成! 最佳 NDCG@3: {study.best_value:.4f}")
        print(f"   最佳参数: {study.best_params}")
        return study.best_params
    
    def _objective(self, trial: optuna.Trial, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> float:
        """Optuna 目标函数"""
        df = df.sort_values('timestamp')
        groups = df.groupby('timestamp').size().tolist()
        X, y = df[feature_cols], df[target_col]
        
        split_idx = int(len(groups) * 0.8)
        train_size = sum(groups[:split_idx])
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        params = {
            'objective': 'lambdarank', 'metric': 'ndcg',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'label_gain': np.arange(max(y) + 1).tolist(),
            'random_state': 42, 'verbosity': -1
        }
        
        model = lgb.LGBMRanker(**params)
        model.fit(X_train, y_train, group=groups[:split_idx],
                 eval_set=[(X_test, y_test)], eval_group=[groups[split_idx:]],
                 eval_at=[3], callbacks=[lgb.early_stopping(20, verbose=False)])
        
        return model.best_score_['valid_0']['ndcg@3']


class SignalClassifierTrainer:
    def __init__(self, model_name: str = "Universal_Signal_Classifier"):
        self.model_name = model_name
        self.model = None

    def train(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        target_col: str,
        purge_periods: int = 5,
        use_time_decay: bool = True,
        decay_half_life_days: int = 90
    ) -> dict:
        """
        使用稳健训练方法 (Purged CV + 样本加权)
        
        Args:
            df: 训练数据
            feature_cols: 特征列
            target_col: 目标列
            purge_periods: 样本清除周期数 (防止信息泄露)
            use_time_decay: 是否使用时间衰减权重
            decay_half_life_days: 时间衰减半衰期 (天)
            
        Returns:
            训练结果字典 (包含 cv_scores, mean_f1, std_f1)
        """
        config = RobustTrainConfig(
            purge_periods=purge_periods,
            use_time_decay=use_time_decay,
            decay_half_life_days=decay_half_life_days
        )
        
        robust_trainer = RobustTrainer(config)
        self.model, results = robust_trainer.train_classifier(
            df, feature_cols, target_col
        )
        
        return results
    
    # 保留别名以兼容旧代码
    train_robust = train

    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, path)
            print(f"Model saved to {path}")

    def load(self, path: str):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model:
            return self.model.predict(X)
        return np.array([])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model:
            return self.model.predict_proba(X)
        return np.array([])
    
    def optimize(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, n_trials: int = 50) -> dict:
        """使用 Optuna 优化超参数"""
        print(f"\n🔍 开始 Optuna 超参数优化 (L3 Classifier)... 试验次数: {n_trials}")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._objective(trial, df, feature_cols, target_col),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        print(f"\n✅ 优化完成! 最佳 F1-Score: {study.best_value:.4f}")
        print(f"   最佳参数: {study.best_params}")
        return study.best_params
    
    def _objective(self, trial: optuna.Trial, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> float:
        """Optuna 目标函数"""
        X, y = df[feature_cols], df[target_col]
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42, 'verbosity': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                 callbacks=[lgb.early_stopping(20, verbose=False)])
        
        # 返回 F1-Score (宏平均)
        from sklearn.metrics import f1_score
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred, average='macro')


class SklearnClassifierTrainer:
    def __init__(self, model_type: str = "rf"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        if model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = LogisticRegression(max_iter=1000)
            
    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        # 简单的时间序列切分
        df = df.sort_values("timestamp")
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]
        
        self.model.fit(X_train, y_train)
        
        # 评估
        score = self.model.score(X_test, y_test)
        print(f"Sklearn Model ({type(self.model).__name__}) Accuracy: {score:.4f}")
        return score

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)
        return self.model
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model:
            return self.model.predict_proba(X)
        return np.array([])

class RiskModelTrainer:
    """
    L4 风控模型训练器 - 用于预测止盈止损点位
    """
    def __init__(self):
        self.models = {}
    
    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, model_name: str) -> None:
        """
        训练单个风控回归模型
        
        参数:
            df: 训练数据
            feature_cols: 特征列
            target_col: 目标列 (止盈/止损百分比)
            model_name: 模型名称 (tp_long, sl_long, tp_short, sl_short)
        """
        # 1. 准备数据
        X = df[feature_cols]
        y = df[target_col]
        
        # 2. 时间序列划分
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training {model_name} model... Samples: {len(X_train)} train, {len(X_test)} test")
        print(f"Target range: [{y_train.min():.2%}, {y_train.max():.2%}]")
        
        # 3. 模型配置
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=150,
            random_state=42,
            verbosity=-1
        )
        
        # 4. 训练
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # 5. 评估
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
        
        print(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        
        self.models[model_name] = model
    
    def save(self, path: str) -> None:
        """保存当前训练的模型"""
        if self.models:
            # 只保存最后一个训练的模型
            model_name = list(self.models.keys())[-1]
            joblib.dump(self.models[model_name], path)
            print(f"Model saved to {path}")
    
    def load(self, path: str, model_name: str):
        """加载模型"""
        model = joblib.load(path)
        self.models[model_name] = model
        print(f"Model loaded from {path}")
        return model
    
    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """预测"""
        if model_name in self.models:
            return self.models[model_name].predict(X)
        return np.array([])
