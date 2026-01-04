import os
import sys
import joblib
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from data.provider import DataProvider
from features.builder import FeatureBuilder

# 加载环境变量
load_dotenv()

def predict_tomorrow():
    symbol = "QQQ"
    timeframe = TimeFrame.Day # 默认值
    
    # 尝试从命令行参数获取周期 (如: python predict.py 15m)
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '1d':
            timeframe = TimeFrame.Day
        elif arg == '1h':
            timeframe = TimeFrame.Hour
        elif arg.endswith('m'):
            try:
                mins = int(arg.replace('m', ''))
                timeframe = TimeFrame(mins, TimeFrameUnit.Minute)
            except ValueError:
                pass
    
    # 转换周期为业界规范字符串 (如 1d, 15m)
    tf_str = DataProvider.get_tf_string(timeframe)
    model_path = f"output/{symbol}_{tf_str}_lgbm.joblib"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}。")
        print(f"请先运行: python main.py {tf_str}")
        return

    print(f"正在获取 {symbol} ({tf_str}) 的最新数据进行预测...")
    
    try:
        # 1. 准备数据
        provider = DataProvider()
        end_date = datetime.now()
        # 根据周期获取足够的历史数据来计算指标
        days_back = 60 if timeframe.unit == TimeFrameUnit.Day else 5
        start_date = end_date - timedelta(days=days_back)
        
        df = provider.fetch_bars(symbol, timeframe, start_date, end_date)
        
        # 2. 特征工程 (is_training=False 保证保留最后一行)
        builder = FeatureBuilder()
        df_features = builder.add_all_features(df, is_training=False)
        
        # 获取最后一行（最新的数据点）
        latest_data = df_features.tail(1)
        latest_time = latest_data['timestamp'].iloc[0]
        
        # 3. 加载模型
        model = joblib.load(model_path)
        
        # 4. 定义特征列 (必须与训练时一致)
        feature_cols = [
            'return_1d', 'return_5d', 'ma_5', 'ma_20', 
            'ma_ratio', 'rsi', 'volatility_20d',
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'volume_ratio', 'volume_change'
        ]
        
        X_latest = latest_data[feature_cols]
        
        # 5. 执行预测
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0]
        
        print("\n" + "="*40)
        print(f"预测标的: {symbol}")
        print(f"最新数据时间: {latest_time}")
        print("-" * 40)
        
        if prediction == 1:
            print(f"预测结果: 📈 [上涨]")
            print(f"上涨概率: {probability[1]:.2%}")
        else:
            print(f"预测结果: 📉 [下跌]")
            print(f"下跌概率: {probability[0]:.2%}")
        
        print("="*40)
        print("注意: 预测结果仅供参考，不构成投资建议。")

    except Exception as e:
        print(f"预测过程中出错: {e}")

if __name__ == "__main__":
    predict_tomorrow()
