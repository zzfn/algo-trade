import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import RiskModelTrainer
from models.constants import get_feature_columns, L2_SYMBOLS
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

def train_l4_model():
    """
    训练 L4 收益预测模型 (包含 L1 宏观特征)。
    
    新版 L4 预测未来 5 周期收益率,用于动态仓位分配。
    现在包含 L1 宏观特征,让模型自动学习市场环境对收益的影响。
    """
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 1. 获取 L1 宏观数据 (日线)
    print("=" * 60)
    print("Step 1: Fetching L1 macro data...")
    print("=" * 60)
    
    end_date = datetime(2024, 12, 31)
    l1_start = end_date - timedelta(days=365 * 2)  # L1 需要更长历史
    
    from features.macro import L1FeatureBuilder
    from models.constants import L1_SYMBOLS
    
    l1_builder = L1FeatureBuilder()
    l1_dict = {}
    for sym in L1_SYMBOLS:
        df = provider.fetch_bars(sym, TimeFrame.Day, l1_start, end_date)
        l1_dict[sym] = df
        print(f"  Loaded {len(df)} days for {sym}")
    
    l1_features = l1_builder.build_l1_features(l1_dict)
    print(f"  L1 features built: {len(l1_features)} rows")
    print(f"  L1 columns: {[c for c in l1_features.columns if c != 'timestamp']}")
    
    # 2. 获取 L4 技术数据 (小时线)
    print("\n" + "=" * 60)
    print("Step 2: Fetching L4 technical data...")
    print("=" * 60)
    
    start_date = end_date - timedelta(days=365)
    symbols = L2_SYMBOLS
    print(f"  Fetching data for {len(symbols)} stocks...")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame.Hour, start_date, end_date)
    print(f"  Raw data rows: {len(df_raw)}")
    
    # 3. 构建技术特征
    print("\n" + "=" * 60)
    print("Step 3: Building technical features...")
    print("=" * 60)
    
    df = builder.add_all_features(df_raw, is_training=False)
    print(f"  Technical features added: {len(df)} rows")
    
    # 4. 合并 L1 宏观特征
    print("\n" + "=" * 60)
    print("Step 4: Merging L1 macro features...")
    print("=" * 60)
    
    df = builder.merge_l1_features(df, l1_features)
    print(f"  L1 features merged: {len(df)} rows")
    
    # 验证 L1 特征已添加
    l1_cols = [c for c in df.columns if c.startswith('spy_') or c.startswith('vixy_') or c.startswith('tlt_')]
    print(f"  L1 macro columns in dataset: {l1_cols}")
    
    # 5. 添加收益目标
    print("\n" + "=" * 60)
    print("Step 5: Adding return targets...")
    print("=" * 60)
    
    df = builder.add_return_target(df, horizon=5)
    
    # 删除 NaN
    df = df.dropna()
    print(f"  Valid training samples: {len(df)}")
    
    # 打印收益分布
    print(f"\n  Target return distribution:")
    print(f"    Mean:   {df['target_future_return'].mean():.4%}")
    print(f"    Std:    {df['target_future_return'].std():.4%}")
    print(f"    Min:    {df['target_future_return'].min():.4%}")
    print(f"    Max:    {df['target_future_return'].max():.4%}")
    
    # 6. 准备特征列 (包含 L1 特征)
    print("\n" + "=" * 60)
    print("Step 6: Preparing features for training...")
    print("=" * 60)
    
    feature_cols = get_feature_columns(df)
    print(f"  Total features: {len(feature_cols)}")
    print(f"  L1 macro features: {len([c for c in feature_cols if c in l1_cols])}")
    print(f"  Technical features: {len(feature_cols) - len([c for c in feature_cols if c in l1_cols])}")
    
    # 7. 训练模型
    print("\n" + "=" * 60)
    print("Step 7: Training L4 Return Prediction model...")
    print("=" * 60)
    
    trainer = RiskModelTrainer()
    trainer.train(df, feature_cols, 'target_future_return', 'return_predictor')
    trainer.save("models/artifacts/l4_return_predictor.joblib")
    
    print("\n" + "=" * 60)
    print("✅ L4 Return Prediction model training complete!")
    print("=" * 60)
    print("  Model saved to: models/artifacts/l4_return_predictor.joblib")
    print("  Features include: Technical indicators + L1 macro features")
    print("  Model will automatically adjust predictions based on market environment")
    print("=" * 60)

if __name__ == "__main__":
    train_l4_model()

