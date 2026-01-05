
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from models.trainer import RankingModelTrainer, SignalClassifierTrainer, RiskModelTrainer, SklearnClassifierTrainer

def inspect_models():
    print("📊 模型特征重要性分析 (Feature Importance Analysis)\n")
    
    # 定义模型加载器和路径
    models = [
        ("L1 Market Timing", "models/artifacts/l1_market_timing.joblib", SklearnClassifierTrainer()),
        ("L2 Stock Selection", "models/artifacts/l2_stock_selection.joblib", RankingModelTrainer()),
        ("L3 Execution", "models/artifacts/l3_execution.joblib", SignalClassifierTrainer()),
        ("L4 Risk Predictor", "models/artifacts/l4_return_predictor.joblib", RiskModelTrainer())
    ]
    
    # 翻译字典 (English -> Chinese Description)
    desc_map = {
        'spy_dist_ma200': 'SPY 距离 MA200 (大趋势)',
        'vixy_level': 'VIXY 恐慌指数 (风险)',
        'tlt_return_5d': 'TLT 美债收益 (流动性)',
        'vixy_change_1d': 'VIXY 变化率',
        'spy_return_1d': 'SPY 单日涨幅',
        'volatility_20d': '20日波动率 (活跃度)',
        'adx': 'ADX 趋势强度',
        'volume_ma_5': '5日均量 (流动性)',
        'bb_width': '布林带宽 (变盘点)',
        'swing_low': 'SMC 前低结构 (支撑)',
        'swing_high': 'SMC 前高结构 (阻力)',
        'return_5p': '5日涨幅 (短期趋势)',
        'return_1p': '1日涨幅 (即时动量)',
        'bb_lower_rel': '布林下轨距离 (超卖)',
        'bb_upper_rel': '布林上轨距离 (超买)',
        'upper_wick_rel': '上影线 (抛压)',
        'lower_wick_rel': '下影线 (承接)',
        'candle_range_rel': 'K线长度 (动能)',
        'volume_ratio': '量比 (放量)',
        'volume_change': '成交量变化',
        'wick_ratio': '影线比例 (形态)',
        'local_high': '局部高点',
        'local_low': '局部低点',
        'bos_up': 'SMC 向上破位 (BOS)',
        'bos_down': 'SMC 向下破位 (BOS)',
        'fvg_size_rel': 'FVG 缺口大小',
        'ob_bullish': '看涨订单块 (OB)',
        'ob_bearish': '看跌订单块 (OB)',
        'ema_20_rel': 'EMA20 乖离率',
        'is_pin_bar': 'Pin Bar 形态',
        'is_engulfing': '吞没形态',
        'shakeout_bull': 'SMC 多头洗盘',
        'shakeout_bear': 'SMC 空头洗盘',
        'displacement': '大阳/阴线 (动能)'
    }

    for name, path, trainer in models:
        print(f"--- {name} ---")
        try:
            if "l4" in path:
                loaded = joblib.load(path)
                model = loaded.get('model') if isinstance(loaded, dict) else loaded
            else:
                trainer.model = joblib.load(path)
                model = trainer.model

            # 提取特征重要性
            importances = None
            feature_names = None
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'): 
                importances = np.abs(model.coef_[0])
            
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            
            if importances is not None:
                if feature_names is None:
                    feature_names = [f"Feature_{i}" for i in range(len(importances))]
                
                # 构建 Description 列
                descriptions = [desc_map.get(f, '-') for f in feature_names]

                # 创建 DataFrame
                df_imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Description': descriptions,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # 打印 Top 12 (为了看更多因子)
                print(df_imp.head(12).to_string(index=False))
                print("\n")
            else:
                print("模型不支持特征重要性提取。\n")
                
        except Exception as e:
            print(f"无法加载或分析模型: {e}\n")

if __name__ == "__main__":
    inspect_models()
