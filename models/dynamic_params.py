"""
动态参数加载器

自动从 L5 模型加载最优参数,如果 L5 不可用则使用默认值
"""

import numpy as np
from pathlib import Path

class DynamicParams:
    """动态参数管理器"""
    
    def __init__(self):
        self._l5_model = None
        self._load_l5_model()
    
    def _load_l5_model(self):
        """尝试加载 L5 模型"""
        try:
            from models.meta_strategy import MetaStrategyModel
            model_path = Path('models/artifacts/l5_meta_strategy.joblib')
            
            if model_path.exists():
                self._l5_model = MetaStrategyModel()
                self._l5_model.load(str(model_path))
                print("✅ L5 元策略模型已加载,将使用动态参数")
            else:
                print("ℹ️  L5 模型未找到,使用默认参数")
        except Exception as e:
            print(f"⚠️  L5 模型加载失败: {e},使用默认参数")
            self._l5_model = None
    
    def get_params(self, market_features=None):
        """
        获取当前最优参数
        
        Args:
            market_features: 市场特征字典,如果为 None 则使用默认市场特征
            
        Returns:
            参数字典
        """
        # 如果 L5 可用,使用动态预测
        if self._l5_model is not None:
            if market_features is None:
                # 使用默认市场特征
                market_features = {
                    'spy_return_1d': 0.01,
                    'spy_volatility': 0.02,
                    'vixy_level': 16.0,
                    'market_trend': 1,
                    'recent_volatility': 0.015
                }
            
            try:
                params = self._l5_model.predict_optimal_params(market_features)
                
                # 清理 numpy 类型
                return {
                    'signal_threshold': float(params['signal_threshold']),
                    'top_n_trades': int(params['top_n_trades']),
                    'l1_risk_factor': float(params['l1_risk_factor'])
                }
            except Exception as e:
                print(f"⚠️  L5 预测失败: {e},使用默认参数")
        
        # 回退到默认参数
        return {
            'signal_threshold': 0.517,
            'top_n_trades': 2,
            'l1_risk_factor': 0.633
        }

# 全局单例
_dynamic_params = DynamicParams()

def get_dynamic_params(market_features=None):
    """获取动态参数的便捷函数"""
    return _dynamic_params.get_params(market_features)

# 导出常量(向后兼容)
_params = get_dynamic_params()
SIGNAL_THRESHOLD = _params['signal_threshold']
TOP_N_TRADES = _params['top_n_trades']
L1_RISK_FACTOR = _params['l1_risk_factor']
