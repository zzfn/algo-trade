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
    parser = argparse.ArgumentParser(description="QQQ 价格涨跌预测工具")
    parser.add_argument("timeframe", nargs="?", default="1d", help="预测周期 (如 1d, 15m, 1h)")
    parser.add_argument("--date", help="指定预测日期 (格式: YYYY-MM-DD), 不传则预测最新数据")
    
    args = parser.parse_args()
    
    symbol = "QQQ"
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

    model_path = f"output/{symbol}_{tf_str}_lgbm.joblib"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}。")
        print(f"请先运行: python main.py {tf_str}")
        return

    print(f"正在获取 {symbol} ({tf_str}) 的历史数据进行分析...")
    
    try:
        provider = DataProvider()
        
        if args.date:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
            # 如果是指定日期，我们需要拉取到指定日期后的几条数据，以便验证实际结果
            end_date = target_date + timedelta(days=7) 
            start_date = target_date - timedelta(days=60)
        else:
            target_date = None
            end_date = datetime.now()
            days_back = 60 if timeframe.unit == TimeFrameUnit.Day else 5
            start_date = end_date - timedelta(days=days_back)
        
        df = provider.fetch_bars(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            print("错误: 未获取到数据。")
            return

        # 2. 特征工程
        builder = FeatureBuilder()
        # is_training=False 会保留最后一行
        df_features = builder.add_all_features(df, is_training=False)
        
        if target_date:
            # 找到最接近 target_date 的那条记录
            df_features['date_diff'] = (df_features['timestamp'].dt.tz_localize(None) - target_date).abs()
            latest_data = df_features.sort_values('date_diff').head(1)
        else:
            # 获取最后一行
            latest_data = df_features.tail(1)
            
        latest_time = latest_data['timestamp'].iloc[0]
        actual_price = latest_data['close'].iloc[0]
        
        # 3. 加载模型
        model = joblib.load(model_path)
        
        # 4. 定义特征列 (必须与训练时一致)
        feature_cols = [
            'return_1d', 'return_5d', 'ma_5', 'ma_20', 
            'ma_ratio', 'rsi', 'volatility_20d',
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'volume_ratio', 'volume_change',
            'wick_ratio', 'is_pin_bar', 'is_engulfing',
            'fvg_up', 'fvg_down', 'displacement'
        ]
        
        X_latest = latest_data[feature_cols]
        
        # 5. 执行预测
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0]
        
        print("\n" + "="*50)
        print(f"预测标的: {symbol} ({tf_str})")
        print(f"分析基准时间: {latest_time}")
        print(f"当前收盘价: {actual_price:.2f}")
        print("-" * 50)
        
        res_str = "📈 [上涨]" if prediction == 1 else "📉 [下跌]"
        prob_val = probability[1] if prediction == 1 else probability[0]
        print(f"预测结果: {res_str}")
        print(f"预测概率: {prob_val:.2%}")
        
        # 6. 验证实际结果 (如果数据中有下一条记录)
        future_data = df_features[df_features['timestamp'] > latest_time].head(1)
        if not future_data.empty:
            next_time = future_data['timestamp'].iloc[0]
            next_close = future_data['close'].iloc[0]
            actual_move = 1 if next_close > actual_price else 0
            actual_str = "📈 [上涨]" if actual_move == 1 else "📉 [下跌]"
            
            print("-" * 50)
            print(f"实际结果时间: {next_time}")
            print(f"实际收盘价: {next_close:.2f}")
            print(f"实际走势: {actual_str}")
            
            if prediction == actual_move:
                print("验证结论: ✅ [预测正确]")
            else:
                print("验证结论: ❌ [预测错误]")
        else:
            print("-" * 50)
            print("验证结论: ⏳ [待市场验证] (这是最新一条数据，尚无后续行情)")
        
        print("="*50)

    except Exception as e:
        print(f"预测过程中出错: {e}")

if __name__ == "__main__":
    run_prediction()
