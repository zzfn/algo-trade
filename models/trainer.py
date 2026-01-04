import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple

class RankingModelTrainer:
    def __init__(self, model_name: str = "Mag7_Ranker"):
        self.model_name = model_name
        self.model = None

    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> None:
        """
        使用 LightGBM Ranker (LambdaRank) 进行模型训练。
        数据需要按 timestamp 排序，以便正确计算 group。
        """
        # 1. 确保按时间排序
        df = df.sort_values('timestamp')
        
        # 2. 计算分组 (每秒/每个时间点有多少个标的)
        groups = df.groupby('timestamp').size().tolist()
        
        # 3. 准备数据
        X = df[feature_cols]
        y = df[target_col]
        
        # 4. 时间序列划分 (80% 训练, 20% 测试)
        split_idx = int(len(groups) * 0.8)
        train_groups = groups[:split_idx]
        test_groups = groups[split_idx:]
        
        train_size = sum(train_groups)
        
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        print(f"正在训练排序模型... 总组数: {len(groups)}, 训练组数: {len(train_groups)}, 测试组数: {len(test_groups)}")
        
        # 5. 模型配置
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            importance_type='gain',
            label_gain=np.arange(max(y) + 1).tolist(), # 针对打分的增益设置
            random_state=42,
            verbosity=-1
        )
        
        # 6. 执行训练
        self.model.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_test, y_test)],
            eval_group=[test_groups],
            eval_at=[1, 3] # 关注前 1 名和前 3 名的排序情况
        )
        
        print("模型训练完成。")

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

class SignalClassifierTrainer:
    def __init__(self, model_name: str = "Universal_Signal_Classifier"):
        self.model_name = model_name
        self.model = None

    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> None:
        """
        使用 LightGBM Classifier 进行二分类训练。
        """
        # 1. 准备数据
        X = df[feature_cols]
        y = df[target_col]
        
        # 2. 时间序列划分 (按行，因为现在不需要 group)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"正在训练信号分类模型... 总样本: {len(df)}, 训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        print(f"信号分布: \n{y_train.value_counts(normalize=True)}")
        
        # 3. 模型配置
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
            random_state=42,
            verbosity=-1
        )
        
        # 4. 执行训练
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)]
        )

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
