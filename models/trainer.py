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

    def save_model(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, path)
            print(f"模型已保存至 {path}")
        else:
            print("错误: 没有可保存的模型。")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model:
            return self.model.predict(X)
        return np.array([])
