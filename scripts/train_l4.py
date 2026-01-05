import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import RiskModelTrainer
from models.constants import get_feature_columns
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import os

def train_l4_model():
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 1. 获取 1 年的小时线数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC']
    print(f"Fetching data for {len(symbols)} stocks for L4 risk management...")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame.Hour, start_date, end_date)
    print(f"Raw data rows: {len(df_raw)}")
    
    # 2. 构建特征和风控标签
    print("Building features and risk targets...")
    df = builder.add_all_features(df_raw, is_training=False)  # 不需要 classification target
    df = builder.add_risk_targets(df, horizon=20)  # 添加止盈止损标签
    
    # 删除 NaN
    df = df.dropna()
    print(f"Valid training samples: {len(df)}")
    
    # 3. 准备特征列
    feature_cols = get_feature_columns(df)
    
    print(f"Training L4 Risk Management models with {len(feature_cols)} features.")
    
    # 4. 训练 4 个模型
    trainer = RiskModelTrainer()
    
    # 做多止盈
    print("\n[1/4] Training Long Take-Profit model...")
    trainer.train(df, feature_cols, 'target_tp_long_pct', 'tp_long')
    trainer.save("models/artifacts/l4_risk_tp_long.joblib")
    
    # 做多止损
    print("\n[2/4] Training Long Stop-Loss model...")
    trainer.train(df, feature_cols, 'target_sl_long_pct', 'sl_long')
    trainer.save("models/artifacts/l4_risk_sl_long.joblib")
    
    # 做空止盈
    print("\n[3/4] Training Short Take-Profit model...")
    trainer.train(df, feature_cols, 'target_tp_short_pct', 'tp_short')
    trainer.save("models/artifacts/l4_risk_tp_short.joblib")
    
    # 做空止损
    print("\n[4/4] Training Short Stop-Loss model...")
    trainer.train(df, feature_cols, 'target_sl_short_pct', 'sl_short')
    trainer.save("models/artifacts/l4_risk_sl_short.joblib")
    
    print("\nL4 Risk Management models training complete.")

if __name__ == "__main__":
    train_l4_model()
