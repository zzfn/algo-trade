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
    训练 L4 收益预测模型。
    
    新版 L4 预测未来 5 周期收益率，用于动态仓位分配。
    止盈止损改为使用 SMC 规则引擎 (models/smc_rules.py)。
    """
    load_dotenv()
    provider = DataProvider()
    builder = FeatureBuilder()
    
    # 1. 获取 1 年的小时线数据
    # 截止日期固定为 2024-12-31，2025 年数据用于样本外验证
    # L4: 收益预测模型 (Return Prediction) -> 动态仓位; SMC 风控用于 TP/SL
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=365)
    
    # 使用统一的标的池
    symbols = L2_SYMBOLS
    print(f"Fetching data for {len(symbols)} stocks for L4 return prediction...")
    
    df_raw = provider.fetch_bars(symbols, TimeFrame.Hour, start_date, end_date)
    print(f"Raw data rows: {len(df_raw)}")
    
    # 2. 构建特征和收益标签
    print("Building features and return targets...")
    df = builder.add_all_features(df_raw, is_training=False)
    df = builder.add_return_target(df, horizon=5)  # 预测未来 5 周期收益率
    
    # 删除 NaN
    df = df.dropna()
    print(f"Valid training samples: {len(df)}")
    
    # 打印收益分布
    print(f"\nTarget return distribution:")
    print(f"  Mean:   {df['target_future_return'].mean():.4%}")
    print(f"  Std:    {df['target_future_return'].std():.4%}")
    print(f"  Min:    {df['target_future_return'].min():.4%}")
    print(f"  Max:    {df['target_future_return'].max():.4%}")
    
    # 3. 准备特征列
    feature_cols = get_feature_columns(df)
    
    print(f"\nTraining L4 Return Prediction model with {len(feature_cols)} features.")
    
    # 4. 训练单个收益预测模型
    trainer = RiskModelTrainer()
    trainer.train(df, feature_cols, 'target_future_return', 'return_predictor')
    trainer.save("models/artifacts/l4_return_predictor.joblib")
    
    print("\n✅ L4 Return Prediction model training complete.")
    print("   Model saved to: models/artifacts/l4_return_predictor.joblib")
    print("   Note: Stop-loss/Take-profit now uses SMC rules (models/smc_rules.py)")

if __name__ == "__main__":
    train_l4_model()

