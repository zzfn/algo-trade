import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import RankingModelTrainer
from models.constants import get_feature_columns
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import os

def train_l2_model():
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 1. 获取 1 年的小时线数据
    # 截止日期固定为 2024-12-31，2025 年数据用于样本外验证
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=365)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC']
    print(f"Fetching data for {len(symbols)} stocks for L2 ranking...")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame.Hour, start_date, end_date)
    print(f"Raw data rows: {len(df_raw)}")
    
    # 2. 构建特征和排名标签
    print("Building features and rank targets...")
    df = builder.add_all_features(df_raw, is_training=True)
    df = builder.add_rank_target(df, horizon=5)
    
    # 3. 准备特征列
    feature_cols = get_feature_columns(df)
    
    print(f"Training L2 Ranker with {len(feature_cols)} features.")
    
    # 4. 训练模型
    trainer = RankingModelTrainer()
    trainer.train(df, feature_cols, 'target_rank')
    
    # 5. 保存
    trainer.save("models/artifacts/l2_stock_selection.joblib")
    print("L2 Stock Selection model training complete.")

if __name__ == "__main__":
    train_l2_model()
