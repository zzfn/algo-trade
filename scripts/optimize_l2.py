import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from training.trainer import RankingModelTrainer
from config.settings import get_feature_columns
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
import json
from pathlib import Path

def optimize_l2():
    """优化 L2 选股排序模型"""
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 获取数据
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=365)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC']
    
    print(f"📊 获取数据...")
    df_raw = provider.fetch_bars(symbols, TimeFrame(15, TimeFrameUnit.Minute), start_date, end_date)
    print(f"   原始数据: {len(df_raw)} 行")
    
    # 构建特征
    print(f"🔧 构建特征...")
    df = builder.add_all_features(df_raw, is_training=True)
    df = builder.add_rank_target(df, horizon=4)
    feature_cols = get_feature_columns(df)
    print(f"   特征数量: {len(feature_cols)}")
    
    # 优化
    trainer = RankingModelTrainer()
    best_params = trainer.optimize(df, feature_cols, 'target_rank', n_trials=50)
    
    # 保存最佳参数
    params_file = Path('config/best_params.json')
    if params_file.exists():
        with open(params_file) as f:
            all_params = json.load(f)
    else:
        all_params = {}
    
    all_params['l2'] = best_params
    
    with open(params_file, 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print(f"\n💾 最佳参数已保存到: {params_file}")
    print(f"   参数: {best_params}")

if __name__ == "__main__":
    optimize_l2()
