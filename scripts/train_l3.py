import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import SignalClassifierTrainer
from models.constants import get_feature_columns, L2_SYMBOLS
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

def train_l3_model():
    """训练 L3 趋势确认模型 (使用 Purged CV + 样本加权)"""
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 1. 获取数据
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=60)
    
    symbols = L2_SYMBOLS
    print(f"Fetching 1m data for {len(symbols)} stocks...")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame.Minute, start_date, end_date)
    print(f"Raw data rows: {len(df_raw)}")
    
    # 2. 构建特征 (预处理已集成在 add_all_features 中)
    print("Building features...")
    df = builder.add_all_features(df_raw, is_training=False)
    df = builder.add_classification_target(df, horizon=15, threshold=0.003)
    # dropna 已在 add_all_features(is_training=True) 中处理,这里需要手动调用
    df = df.dropna()
    
    feature_cols = get_feature_columns(df)
    print(f"Training with {len(feature_cols)} features.")
    
    # 3. 稳健训练 (purge_periods=15 与 horizon 相同)
    trainer = SignalClassifierTrainer(model_name="L3_Trend_Model")
    results = trainer.train_robust(df, feature_cols, 'target_signal', purge_periods=15)
    print(f"\n📊 结果: F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    
    # 4. 保存
    trainer.save("models/artifacts/l3_execution.joblib")
    print("✅ 完成!")

if __name__ == "__main__":
    train_l3_model()
