# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import RankingModelTrainer
from models.constants import get_feature_columns
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

def train_l2_model():
    """训练 L2 股票排序模型 (使用 Purged CV + 样本加权)"""
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 1. 获取数据 (5min 频率)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90) # 5min 数据量大，先回溯 90 天
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC']
    print(f"Fetching 5min data for {len(symbols)} stocks...")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame(5, TimeFrameUnit.Minute), start_date, end_date)
    print(f"Raw data rows: {len(df_raw)}")
    
    # 2. 构建特征
    print("Building features...")
    df = builder.add_all_features(df_raw, is_training=True)
    # 设置 1 小时后的排名 (5min * 12)
    df = builder.add_rank_target(df, horizon=12)
    
    feature_cols = get_feature_columns(df)
    print(f"Training with {len(feature_cols)} features.")
    
    # 3. 稳健训练
    trainer = RankingModelTrainer()
    results = trainer.train_robust(df, feature_cols, 'target_rank')
    print(f"\n📊 结果: NDCG@3 = {results['mean_ndcg']:.4f} ± {results['std_ndcg']:.4f}")
    
    # 4. 保存
    trainer.save("models/artifacts/l2_stock_selection.joblib")
    print("✅ 完成!")

if __name__ == "__main__":
    train_l2_model()
