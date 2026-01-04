import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import RankingModelTrainer

# 加载环境变量
load_dotenv()

def main():
    # 标的池: 指数 + 七姐妹
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    
    # 解析命令行参数
    tf_arg = sys.argv[1] if len(sys.argv) > 1 else "1d"
    
    if tf_arg == '1d':
        timeframe = TimeFrame.Day
        start_date = datetime.now() - timedelta(days=365 * 5) # 5年
    elif tf_arg == '1h':
        timeframe = TimeFrame.Hour
        start_date = datetime.now() - timedelta(days=365) # 1年
    elif tf_arg == '15m':
        timeframe = TimeFrame(15, TimeFrameUnit.Minute)
        start_date = datetime.now() - timedelta(days=60) # 60天
    else:
        print(f"不支持的周期: {tf_arg}。默认使用 1d。")
        timeframe = TimeFrame.Day
        start_date = datetime.now() - timedelta(days=365 * 5)

    tf_str = DataProvider.get_tf_string(timeframe)
    print(f"正在启动多标的排序训练流水线 ({tf_str})...")

    try:
        # 1. 获取数据
        provider = DataProvider()
        df_raw = provider.fetch_bars(symbols, timeframe, start_date)
        print(f"成功获取 {len(df_raw)} 条记录，涉及 {len(df_raw['symbol'].unique())} 个标的。")

        # 2. 特征工程
        builder = FeatureBuilder()
        df_features = builder.add_all_features(df_raw, is_training=True)
        print(f"特征生成完毕。数据形状: {df_features.shape}")

        # 3. 准备特征列表
        feature_cols = [
            'return_1d', 'return_5d', 'ma_5', 'ma_20', 
            'ma_ratio', 'rsi', 'volatility_20d',
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'volume_ratio', 'volume_change',
            'wick_ratio', 'is_pin_bar', 'is_engulfing',
            'fvg_up', 'fvg_down', 'displacement'
        ]
        target_col = 'target_rank'

        # 4. 训练模型
        trainer = RankingModelTrainer(model_name=f"Mag7_{tf_str}_Ranker")
        trainer.train(df_features, feature_cols, target_col)

        # 5. 保存模型
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_path = f"{output_dir}/mag7_{tf_str}_ranker.joblib"
        trainer.save_model(model_path)
        
        print(f"流水线执行成功。模型保存在: {model_path}")

    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
