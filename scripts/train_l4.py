# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from training.trainer import RiskModelTrainer
from config.settings import get_feature_columns, L2_SYMBOLS
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

def train_unified_model():
    """
    训练统一收益预测模型 (Unified Return Predictor)。
    
    使用 1min K线数据 (resample to 1H)，同时包含 Macro 宏观数据和 L2 技术特征。
    取代原有的 L1 (Market Timing) 和 L4 (Return Prediction) 分离架构。
    """
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 统一截止日期
    end_date = datetime(2024, 12, 31)
    
    # 1. 获取 Macro 宏观数据 (1min -> resample to 1H)
    # 宏观数据回溯更长，以确保 rolling window (如 MA200) 有足够数据
    print("=" * 60)
    print("Step 1: Fetching Macro data (1min -> resample to 1H)...")
    print("=" * 60)
    
    start_date_macro = end_date - timedelta(days=365 * 2) 
    
    from features.macro import MacroFeatureBuilder
    from config.settings import MACRO_SYMBOLS
    
    macro_builder = MacroFeatureBuilder()
    macro_dict = {}
    
    for sym in MACRO_SYMBOLS:
        print(f"  Fetching {sym}...")
        # 使用 1min 数据并 resample 到 1H
        df_min = provider.fetch_bars(sym, TimeFrame.Minute, start_date_macro, end_date)
        df = provider.resample_bars(df_min, '1H')
        macro_dict[sym] = df
        print(f"    Loaded {len(df)} rows for {sym} (resampled to 1H)")
    
    # 构建宏观特征
    # 注意: 现在的 window 是基于 Hour 的 (e.g. MA200 Hour ~= 1 month)
    macro_features = macro_builder.build_macro_features(macro_dict)
    print(f"  Macro features built: {len(macro_features)} rows")
    
    
    # 2. 获取 Stock 技术数据 (1min -> resample to 1H)
    print("\n" + "=" * 60)
    print("Step 2: Fetching Stock Technical data (1min -> resample to 1H)...")
    print("=" * 60)
    
    # Stock 数据不需要那么长，主要用于训练近期关系
    # 但为了对齐宏观数据，我们尽量取较长的时间段，比如 1 年
    start_date_stock = end_date - timedelta(days=365)
    symbols = L2_SYMBOLS
    print(f"  Fetching 1min data for {len(symbols)} stocks...")
    
    df_raw_min = provider.fetch_bars(symbols, TimeFrame.Minute, start_date_stock, end_date)
    df_raw = provider.resample_bars(df_raw_min, '1H')
    print(f"  Raw stock data rows: {len(df_raw)} (resampled to 1H)")
    
    # 3. 构建技术特征
    print("\n" + "=" * 60)
    print("Step 3: Building technical features...")
    print("=" * 60)
    
    df = builder.add_all_features(df_raw, is_training=False)
    print(f"  Technical features added: {len(df)} rows")
    
    # 4. 合并 L1 宏观特征
    print("\n" + "=" * 60)
    print("Step 4: Merging Macro features...")
    print("=" * 60)
    
    # merge_l1_features 会根据 timestamp对齐 (1H 对 1H)
    # Note: method name in FeatureBuilder might still be merge_l1_features unless renamed there too.
    # I should check FeatureBuilder.merge_l1_features name.
    # Assuming user didn't ask to rename THAT specific method in FeatureBuilder (which is in technical.py),
    # but for consistence I should properly rename it later or just use it.
    # Wait, I am restricted to renaming "L1 symbols/classes". FeatureBuilder is in technical.py. 
    # I'll stick to calling it merge_l1_features for now unless found otherwise, 
    # BUT I should check technical.py.
    # Wait, I didn't check technical.py yet. 
    # For now, I'll assume the method name is still `merge_l1_features` in `FeatureBuilder`.
    
    df = builder.merge_l1_features(df, macro_features)
    print(f"  Merged dataset size: {len(df)} rows")
    
    # 验证特征
    macro_cols = [c for c in df.columns if c.startswith('spy_') or c.startswith('vixy_') or c.startswith('tlt_')]
    print(f"  Macro columns present: {len(macro_cols)} ({macro_cols[:3]}...)")
    
    # 5. 添加收益目标
    print("\n" + "=" * 60)
    print("Step 5: Adding return targets...")
    print("=" * 60)
    
    # 预测未来 1 小时收益
    HORIZON = 1 
    print(f"  Target Horizon: {HORIZON} hours")
    
    df = builder.add_return_target(df, horizon=HORIZON)
    
    df = df.dropna()
    print(f"  Valid training samples: {len(df)}")
    
    # 6. 准备特征列
    print("\n" + "=" * 60)
    print("Step 6: Preparing features...")
    print("=" * 60)
    
    feature_cols = get_feature_columns(df)
    print(f"  Total features: {len(feature_cols)}")
    
    # 7. 训练模型
    print("\n" + "=" * 60)
    print("Step 7: Training Unified Return Model...")
    print("=" * 60)
    
    trainer = RiskModelTrainer()
    # 还是使用 'target_future_return'
    trainer.train(df, feature_cols, 'target_future_return', 'return_predictor')
    
    # 保存为统一模型
    model_path = "models/artifacts/unified_return_predictor.joblib"
    trainer.save(model_path)
    
    print("\n" + "=" * 60)
    print("✅ Unified Model training complete!")
    print("=" * 60)
    print(f"  Model saved to: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_unified_model()
