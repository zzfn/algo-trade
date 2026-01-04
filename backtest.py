import os
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from data.provider import DataProvider
from features.builder import FeatureBuilder

# 加载环境变量
load_dotenv()

def run_backtest():
    parser = argparse.ArgumentParser(description="QQQ 策略回测工具")
    parser.add_argument("timeframe", nargs="?", default="1d", help="回测周期 (如 1d, 15m, 1h)")
    parser.add_argument("--days", type=int, default=365, help="回测天数 (默认 365 天)")
    
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
        print(f"错误: 找不到模型文件 {model_path}。请先运行: python main.py {tf_str}")
        return

    print(f"开始对 {symbol} ({tf_str}) 进行回测 (过去 {args.days} 天)...")
    
    try:
        # 1. 获取数据
        provider = DataProvider()
        end_date = datetime.now()
        # 需要多拉取一点历史数据以确保初始特征能计算出来
        start_date = end_date - timedelta(days=args.days + 60)
        
        df = provider.fetch_bars(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            print("错误: 未获取到数据。")
            return

        # 2. 特征工程
        builder = FeatureBuilder()
        # is_training=False 保留最后一行，且我们不 dropna(target)
        df_features = builder.add_all_features(df, is_training=False)
        
        # 过滤出回测目标时段的数据
        backtest_start = end_date - timedelta(days=args.days)
        df_test = df_features[df_features['timestamp'].dt.tz_localize(None) >= backtest_start].copy()
        
        if len(df_test) < 10:
            print("警告: 用于回测的样本量太少。")
            return

        # 3. 加载模型
        model = joblib.load(model_path)
        
        # 4. 特征列
        feature_cols = [
            'return_1d', 'return_5d', 'ma_5', 'ma_20', 
            'ma_ratio', 'rsi', 'volatility_20d',
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'volume_ratio', 'volume_change',
            'wick_ratio', 'is_pin_bar', 'is_engulfing',
            'fvg_up', 'fvg_down', 'displacement'
        ]
        
        # 5. 执行回测模拟
        # 预测下一期的方向
        df_test['prediction'] = model.predict(df_test[feature_cols])
        
        # 计算策略收益
        # 下一期的实际波动
        df_test['next_return'] = df_test['close'].pct_change().shift(-1)
        
        # 策略逻辑：如果预测涨，持有多头；否则观望/不持有 (这里模拟 Read only Long 策略)
        df_test['strategy_return'] = df_test['prediction'] * df_test['next_return']
        
        # 计算累计收益
        df_test['cum_market_return'] = (1 + df_test['next_return']).cumprod()
        df_test['cum_strategy_return'] = (1 + df_test['strategy_return']).cumprod()
        
        # 性能指标
        win_rate = (df_test['prediction'] == (df_test['next_return'] > 0).astype(int)).mean()
        total_market_ret = df_test['cum_market_return'].iloc[-2] - 1 if len(df_test) > 1 else 0
        total_strategy_ret = df_test['cum_strategy_return'].iloc[-2] - 1 if len(df_test) > 1 else 0
        
        # 最大回撤
        roll_max = df_test['cum_strategy_return'].cummax()
        drawdown = df_test['cum_strategy_return'] / roll_max - 1
        max_drawdown = drawdown.min()

        print("\n" + "="*50)
        print(f"回测报告: {symbol} ({tf_str})")
        print(f"时间范围: {df_test['timestamp'].iloc[0]} 至 {df_test['timestamp'].iloc[-1]}")
        print(f"交易总天数/周期数: {len(df_test)}")
        print("-" * 50)
        print(f"模型预测准确率: {win_rate:.2%}")
        print(f"市场累计收益 (Buy & Hold): {total_market_ret:.2%}")
        print(f"策略累计收益 (Model): {total_strategy_ret:.2%}")
        print(f"最大回撤 (Max Drawdown): {max_drawdown:.2%}")
        print("-" * 50)
        
        if total_strategy_ret > total_market_ret:
            print("结论: 🏆 [策略跑赢大盘]")
        else:
            print("结论: 📉 [策略表现逊于大盘]")
        
        print("="*50)

    except Exception as e:
        print(f"回测过程中出错: {e}")

if __name__ == "__main__":
    run_backtest()
