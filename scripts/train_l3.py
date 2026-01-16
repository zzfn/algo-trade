import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from training.trainer import SignalClassifierTrainer
from config.settings import get_feature_columns, L2_SYMBOLS
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

def train_l3_model():
    """训练 L3 趋势确认模型 (使用 5min 频率 + 180天数据 + 1.0% 阈值)"""
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 1. 获取数据 (180天 5min 数据)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    symbols = L2_SYMBOLS
    print(f"Fetching 5min data for {len(symbols)} stocks for 180 days...")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame(5, TimeFrameUnit.Minute), start_date, end_date)
    print(f"Raw data rows: {len(df_raw)}")
    
    # 2. 构建特征
    print("Building features...")
    # is_training=False 因为我们只需要特征，目标标签由 add_classification_target 生成
    df = builder.add_all_features(df_raw, is_training=False)
    
    # 设置 1.0% 阈值，未来 30 分钟 (5min * 6)
    print("Adding classification target (Threshold=1.0%, Horizon=6 bars)...")
    df = builder.add_classification_target(df, horizon=6, threshold=0.01)
    
    df = df.dropna()
    
    feature_cols = get_feature_columns(df)
    print(f"Training with {len(feature_cols)} features.")
    
    # 3. 稳健训练 (由于是 5min 频率，purge_periods 也设为 6)
    trainer = SignalClassifierTrainer(model_name="L3_Trend_Model_5min")
    results = trainer.train_robust(df, feature_cols, 'target_signal', purge_periods=6)
    print(f"\n📊 结果: F1 = {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    
    # 4. 保存
    trainer.save("models/artifacts/l3_execution.joblib")
    print("✅ 完成!")

if __name__ == "__main__":
    train_l3_model()
