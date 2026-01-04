import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.macro import L1FeatureBuilder
from models.trainer import SklearnClassifierTrainer
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

def train_l1_model():
    load_dotenv()
    provider = DataProvider()
    builder = L1FeatureBuilder()
    
    # 1. 获取 2 年以上的历史数据以进行宏观训练
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    
    print("Fetching macro data (SPY, VIXY, TLT)...")
    symbols = ['SPY', 'VIXY', 'TLT']
    df_dict = {}
    for sym in symbols:
        df = provider.fetch_bars(sym, TimeFrame.Day, start_date, end_date)
        df_dict[sym] = df
        print(f"Loaded {len(df)} rows for {sym}")
        
    # 2. 构建特征
    print("Building L1 macro features...")
    df_l1 = builder.build_l1_features(df_dict)
    print(f"Final training rows: {len(df_l1)}")
    
    # 3. 训练模型 (Random Forest)
    feature_cols = ['spy_return_1d', 'spy_dist_ma200', 'vixy_level', 'vixy_change_1d', 'tlt_return_5d']
    target_col = 'target_spy_5d'
    
    print(f"Training L1 model with features: {feature_cols}")
    trainer = SklearnClassifierTrainer(model_type="rf")
    trainer.train(df_l1, feature_cols, target_col)
    
    # 4. 保存
    trainer.save("models/artifacts/l1_market_timing.joblib")
    print("L1 Market Timing model training complete.")

if __name__ == "__main__":
    train_l1_model()
