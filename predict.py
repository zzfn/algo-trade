import os
import sys
import joblib
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from data.provider import DataProvider
from features.builder import FeatureBuilder
import argparse

# 加载环境变量
load_dotenv()

def run_prediction():
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="Mag7 + 指数 排序预测工具")
    parser.add_argument("timeframe", nargs="?", default="1d", help="预测周期 (如 1d, 15m, 1h)")
    parser.add_argument("--date", help="指定历史分析日期 (格式: YYYY-MM-DD 或 'YYYY-MM-DD HH:MM:SS')")
    
    args = parser.parse_args()
    
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    tf_str = args.timeframe.lower()
    
    # 映射周期
    if tf_str == '1d':
        timeframe = TimeFrame.Day
    elif tf_str == '1h':
        timeframe = TimeFrame.Hour
    elif tf_str.endswith('m'):
        try:
            mins = int(tf_str.replace('m', ''))
            timeframe = TimeFrame(mins, TimeFrameUnit.Minute)
        except ValueError:
            timeframe = TimeFrame.Day
    else:
        timeframe = TimeFrame.Day

    model_path = f"output/mag7_{tf_str}_ranker.joblib"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}。")
        print(f"请先运行训练命令 (例如: make train-{tf_str})")
        return

    # 1. 确定时间范围
    if args.date:
        try:
            if len(args.date) > 10:
                target_dt = datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S")
            else:
                target_dt = datetime.strptime(args.date, "%Y-%m-%d")
            
            # 为了计算特征，需要从目标时间往前拉数据
            start_date = target_dt - timedelta(days=60)
            # 往后拉一点点以防万一
            end_date = target_dt + timedelta(days=1)
            prediction_mode_desc = f"历史分析时刻: {target_dt}"
        except ValueError:
            print("错误: 日期格式无效。请使用 YYYY-MM-DD 或 'YYYY-MM-DD HH:MM:SS'")
            return
    else:
        target_dt = None
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        prediction_mode_desc = "最新实时数据分析"

    print(f"正在获取 {len(symbols)} 个标的 ({tf_str}) 数据进行预测 ({prediction_mode_desc})...")
    
    try:
        provider = DataProvider()
        df_raw = provider.fetch_bars(symbols, timeframe, start_date, end_date)
        
        if df_raw.empty:
            print("错误: 未获取到数据。")
            return

        # 2. 特征工程
        builder = FeatureBuilder()
        df_features = builder.add_all_features(df_raw, is_training=False)
        
        # 3. 筛选预测时刻的数据
        if target_dt:
            # 找到最接近 target_dt 的 timestamp
            df_features['dt_diff'] = (df_features['timestamp'] - target_dt).abs()
            closest_ts = df_features.sort_values('dt_diff').iloc[0]['timestamp']
            print(f"匹配到最接近的行情时刻: {closest_ts}")
            latest_data = df_features[df_features['timestamp'] == closest_ts].copy()
        else:
            # 使用最新的一个 timestamp
            latest_ts = df_features['timestamp'].max()
            latest_data = df_features[df_features['timestamp'] == latest_ts].copy()
            
        if latest_data.empty:
            print("错误: 处理后的数据为空。")
            return
            
        analysis_time = latest_data['timestamp'].iloc[0]
        
        # 4. 加载模型
        model = joblib.load(model_path)
        
        # 5. 定义特征列 (必须与训练时一致)
        feature_cols = [
            'return_1d', 'return_5d', 'ma_5', 'ma_20', 
            'ma_ratio', 'rsi', 'volatility_20d',
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'volume_ratio', 'volume_change',
            'wick_ratio', 'is_pin_bar', 'is_engulfing',
            'fvg_up', 'fvg_down', 'displacement'
        ]
        
        # 6. 执行预测 (评分)
        latest_data['score'] = model.predict(latest_data[feature_cols])
        
        # 排序
        results = latest_data[['symbol', 'close', 'score']].sort_values('score', ascending=False)
        
        print("\n" + "="*50)
        print(f"Mag7 排序预测分析 ({tf_str})")
        print(f"分析时刻: {analysis_time}")
        print("-" * 50)
        print(f"{'代码':<8} | {'收盘价格':<10} | {'预测得分':<10}")
        print("-" * 50)
        
        for _, row in results.iterrows():
            print(f"{row['symbol']:<8} | {row['close']:<10.2f} | {row['score']:<10.4f}")
            
        print("-" * 50)
        top_symbol = results.iloc[0]['symbol']
        print(f"👉 当时建议: 优先关注 {top_symbol}")
        print("="*50)

    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_prediction()
