import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import SignalClassifierTrainer
from models.constants import get_feature_columns, L2_SYMBOLS
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

def train_l3_model():
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 1. 获取 60 天的 1 分钟线数据 (约2.3万根K线)
    # 截止日期固定为 2024-12-31
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=60)
    
    # 使用与 L2 相同的标的池，保持模型一致性
    symbols = L2_SYMBOLS
    print(f"Fetching 1m data for {len(symbols)} stocks for L3 trend confirmation...")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame.Minute, start_date, end_date)
    print(f"Raw data rows: {len(df_raw)}")
    
    # 2. 构建特征和分类标签 (短线博弈)
    print("Building features and L3 targets...")
    df = builder.add_all_features(df_raw, is_training=False) # 先跑特征
    # L3 目标：未来 15 根 K 线 (15分钟) 是否有 0.3% 的收益
    df = builder.add_classification_target(df, horizon=15, threshold=0.003)
    df = df.dropna()
    
    # 3. 准备特征列
    feature_cols = get_feature_columns(df)
    
    print(f"Training L3 Classifier with {len(feature_cols)} features.")
    
    # 4. 训练模型
    trainer = SignalClassifierTrainer(model_name="L3_Trend_Model")
    trainer.train(df, feature_cols, 'target_signal')
    
    # 5. 保存
    trainer.save("models/artifacts/l3_execution.joblib")
    print("L3 Trend model training complete.")

if __name__ == "__main__":
    train_l3_model()
