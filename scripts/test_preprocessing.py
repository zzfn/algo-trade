"""
测试数据预处理管道与特征工程的集成
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

def test_preprocessing_integration():
    """测试预处理管道是否正确集成到特征工程中"""
    load_dotenv()
    
    print("=" * 60)
    print("🧪 测试数据预处理管道集成")
    print("=" * 60)
    
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 获取少量测试数据
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=7)  # 只取 7 天数据
    symbols = ['AAPL', 'MSFT']
    
    print(f"\n📊 获取测试数据...")
    print(f"   标的: {symbols}")
    print(f"   时间范围: {start_date.date()} -> {end_date.date()}")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame.Hour, start_date, end_date)
    print(f"   原始数据: {len(df_raw)} 行")
    
    # 测试 1: 使用预处理器
    print(f"\n🔬 测试 1: 启用预处理器")
    df_with_prep = builder.add_all_features(df_raw.copy(), is_training=False, use_preprocessor=True)
    print(f"   处理后数据: {len(df_with_prep)} 行")
    print(f"   新增列: {[c for c in df_with_prep.columns if c.startswith('log_')]}")
    
    # 检查 log returns
    if 'log_return_1p' in df_with_prep.columns:
        print(f"\n   ✅ Log Returns 已计算:")
        print(f"      均值: {df_with_prep['log_return_1p'].mean():.6f}")
        print(f"      标准差: {df_with_prep['log_return_1p'].std():.6f}")
        print(f"      缺失值: {df_with_prep['log_return_1p'].isna().sum()}")
    else:
        print(f"\n   ❌ 错误: log_return_1p 未找到!")
    
    # 测试 2: 不使用预处理器 (对比)
    print(f"\n🔬 测试 2: 禁用预处理器 (对比)")
    df_without_prep = builder.add_all_features(df_raw.copy(), is_training=False, use_preprocessor=False)
    print(f"   处理后数据: {len(df_without_prep)} 行")
    
    # 对比
    print(f"\n📈 对比结果:")
    print(f"   数据行数差异: {len(df_with_prep) - len(df_without_prep)} 行")
    print(f"   (预处理器会移除异常值,所以行数可能减少)")
    
    # 检查特征分布
    print(f"\n📊 特征统计 (启用预处理):")
    feature_cols = [c for c in df_with_prep.columns if c not in 
                   ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    print(f"   总特征数: {len(feature_cols)}")
    print(f"   缺失值统计:")
    missing = df_with_prep[feature_cols].isna().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print(f"      ✅ 无缺失值")
    
    print(f"\n✅ 测试完成!")
    return df_with_prep

if __name__ == "__main__":
    test_preprocessing_integration()
