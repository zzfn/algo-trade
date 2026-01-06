import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import SignalClassifierTrainer
from models.constants import get_feature_columns
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import json
from pathlib import Path

def optimize_l3():
    """优化 L3 趋势确认模型"""
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 获取数据
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=365)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC']
    
    print(f"📊 获取数据...")
    df_raw = provider.fetch_bars(symbols, TimeFrame.Minute, start_date, end_date)
    print(f"   原始数据: {len(df_raw)} 行")
    
    # 构建特征
    print(f"🔧 构建特征...")
    df = builder.add_all_features(df_raw, is_training=True)
    feature_cols = get_feature_columns(df)
    print(f"   特征数量: {len(feature_cols)}")
    print(f"   信号分布: \n{df['target_signal'].value_counts()}")
    
    # 优化
    trainer = SignalClassifierTrainer()
    best_params = trainer.optimize(df, feature_cols, 'target_signal', n_trials=50)
    
    # 保存最佳参数
    params_file = Path('models/best_params.json')
    if params_file.exists():
        with open(params_file) as f:
            all_params = json.load(f)
    else:
        all_params = {}
    
    all_params['l3'] = best_params
    
    with open(params_file, 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print(f"\n💾 最佳参数已保存到: {params_file}")
    print(f"   参数: {best_params}")

if __name__ == "__main__":
    optimize_l3()
