"""
L5 元策略模型

根据市场特征预测最优策略参数
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict
import lightgbm as lgb
from pathlib import Path

class MetaStrategyModel:
    """L5 元策略模型 - 预测最优策略参数"""
    
    def __init__(self):
        self.models = {}
        
    def train(self, df: pd.DataFrame):
        """
        训练元策略模型
        
        Args:
            df: 训练数据,包含市场特征和参数表现
        """
        # 特征列
        feature_cols = [
            'spy_return_1d',
            'spy_volatility', 
            'vixy_level',
            'market_trend',
            'recent_volatility'
        ]
        
        # 目标参数
        target_params = ['signal_threshold', 'top_n_trades']
        
        # 为每个参数训练一个模型
        for param in target_params:
            print(f"\n训练 {param} 预测模型...")
            
            # 找出每个市场环境下的最优参数
            # 按市场特征分组,选择 sharpe 最高的参数
            best_params = df.loc[df.groupby(['spy_return_1d', 'vixy_level'])['sharpe_ratio'].idxmax()]
            
            X = best_params[feature_cols]
            y = best_params[param]
            
            # 训练回归模型
            if param == 'top_n_trades':
                # 整数参数,使用分类
                model = lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=4,  # 2, 3, 4, 5
                    n_estimators=100,
                    random_state=42,
                    verbosity=-1
                )
                # 转换为类别 (2->0, 3->1, 4->2, 5->3)
                y = y - 2
            else:
                # 连续参数,使用回归
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=42,
                    verbosity=-1
                )
            
            model.fit(X, y)
            self.models[param] = model
            
            print(f"   ✅ {param} 模型训练完成")
    
    def predict_optimal_params(self, market_features: Dict) -> Dict:
        """
        预测最优策略参数
        
        Args:
            market_features: 市场特征字典
            
        Returns:
            最优参数字典
        """
        # 转换为 DataFrame
        X = pd.DataFrame([market_features])
        
        # 预测每个参数
        params = {}
        
        for param_name, model in self.models.items():
            pred = model.predict(X)[0]
            
            if param_name == 'top_n_trades':
                # 转换回整数 (0->2, 1->3, 2->3, 3->5)
                params[param_name] = int(pred) + 2
            elif param_name == 'signal_threshold':
                # 限制在合理范围
                params[param_name] = np.clip(pred, 0.35, 0.60)
            else:
                params[param_name] = pred
        
        # 确保包含 l1_risk_factor (固定值，该参数不再由 L5 动态预测)
        if 'l1_risk_factor' not in params:
            params['l1_risk_factor'] = 0.633
        
        return params
    
    def save(self, path: str):
        """保存模型"""
        joblib.dump(self.models, path)
        print(f"L5 元策略模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        self.models = joblib.load(path)
        print(f"L5 元策略模型已加载: {path}")
        return self
