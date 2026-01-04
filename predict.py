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
import pytz

# 加载环境变量
load_dotenv()

def run_prediction():
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="Mag7 + 指数 排序预测工具")
    parser.add_argument("timeframe", nargs="?", default="1h", help="预测周期 (如 1d, 15m, 1h)")
    parser.add_argument("--date", help="指定历史分析日期 (格式: YYYY-MM-DD 或 'YYYY-MM-DD HH:MM:SS')")
    parser.add_argument("--model", help="指定模型文件路径")
    parser.add_argument("--symbols", help="指定分析标的，用逗号分隔 (如 AAPL,TSLA,COIN)")
    
    args = parser.parse_args()
    
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    tf_str = args.timeframe.lower()
    
    # 映射周期
    if tf_str == '1d':
        timeframe = TimeFrame.Day
        bar_duration = timedelta(days=1)
    elif tf_str == '1h':
        timeframe = TimeFrame.Hour
        bar_duration = timedelta(hours=1)
    elif tf_str.endswith('m'):
        try:
            mins = int(tf_str.replace('m', ''))
            timeframe = TimeFrame(mins, TimeFrameUnit.Minute)
            bar_duration = timedelta(minutes=mins)
        except ValueError:
            timeframe = TimeFrame.Day
            bar_duration = timedelta(days=1)
    else:
        timeframe = TimeFrame.Day
        bar_duration = timedelta(days=1)

    if args.model:
        model_path = args.model
    else:
        # 默认使用通用分类模型
        model_path = "models/universal_pa_smc_classifier.joblib"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}。")
        if not args.model:
            print(f"请先运行训练命令 (例如: make train-{tf_str})")
        return

    # 1. 确定时间范围 (使用美东时间)
    ny_tz = pytz.timezone("America/New_York")
    
    if args.date:
        try:
            # 输入日期默认为 ET
            if len(args.date) > 10:
                target_dt = datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S")
            else:
                target_dt = datetime.strptime(args.date, "%Y-%m-%d")
            
            # 为了计算特征，需要往前多拉数据
            start_date = target_dt - timedelta(days=60)
            end_date = target_dt + timedelta(days=1)
            prediction_mode_desc = f"历史分析时刻: {target_dt} ET"
        except ValueError:
            print("错误: 日期格式无效。请使用 YYYY-MM-DD 或 'YYYY-MM-DD HH:MM:SS'")
            return
    else:
        # 实时模式：强制获取当前美东时间
        target_dt = datetime.now(ny_tz).replace(tzinfo=None)
        start_date = target_dt - timedelta(days=60)
        end_date = target_dt + timedelta(days=1)
        prediction_mode_desc = f"最新实时分析 (当前 ET: {target_dt.strftime('%Y-%m-%d %H:%M:%S')})"

    print(f"正在获取 {len(symbols)} 个标的 ({tf_str}) 数据进行预测 (时间标准: 美东时间 ET)...")
    print(f"分析模式: {prediction_mode_desc}")
    
    try:
        provider = DataProvider()
        df_raw = provider.fetch_bars(symbols, timeframe, start_date, end_date)
        
        if df_raw.empty:
            print("错误: 未获取到数据。")
            return

        # 2. 特征工程
        builder = FeatureBuilder()
        df_features = builder.add_all_features(df_raw, is_training=False)
        
        # 3. 筛选预测时刻的数据 (采用点对点逻辑：选择在该时刻前已结束的最后一根 K 线)
        # 规则：timestamp + duration <= target_dt
        df_features['is_complete'] = (df_features['timestamp'] + bar_duration) <= target_dt
        
        complete_bars = df_features[df_features['is_complete'] == True]
        
        if complete_bars.empty:
            # 如果没有完全结束的，退而求其次找最近的一根（可能是正在生成的）
            print("提示: 未找到已完全结束的 K 线，使用最近的一根进行参考。")
            latest_ts = df_features['timestamp'].max()
        else:
            latest_ts = complete_bars['timestamp'].max()
            
        print(f"匹配到分析行情时刻: {latest_ts} (覆盖至 {latest_ts + bar_duration})")
        latest_data = df_features[df_features['timestamp'] == latest_ts].copy()
            
        if latest_data.empty:
            print("错误: 处理后的数据为空。")
            return
            
        analysis_time = latest_data['timestamp'].iloc[0]
        
        # 4. 加载模型
        model = joblib.load(model_path)
        
        # 5. 定义特征列 (自动识别，排除非特征列)
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                        'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                        'max_future_return', 'target_signal', 'dt_diff', 'is_complete']
        feature_cols = [c for c in latest_data.columns if c not in exclude_cols]
        
        print(f"输入特征维度: {len(feature_cols)}")
        
        # 6. 执行预测 (概率)
        # 初始化置信度列
        for col in ['long_p', 'short_p']:
            if col not in latest_data.columns:
                latest_data[col] = 0.0

        # 类别 0: Neutral, 1: Long, 2: Short
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(latest_data[feature_cols])
                if probs.shape[1] >= 3:
                    latest_data['long_p'] = probs[:, 1]
                    latest_data['short_p'] = probs[:, 2]
                else:
                    # 如果只有 2 类 (旧的二分类模型)
                    latest_data['long_p'] = probs[:, 1]
                    latest_data['short_p'] = 0.0
            except Exception as e:
                print(f"警告: 概率预测失败 ({e})，可能是特征不匹配。")
            
            # score 用于主排序逻辑，这里取较大的概率
            latest_data['score'] = latest_data[['long_p', 'short_p']].max(axis=1)
        else:
            try:
                latest_data['score'] = model.predict(latest_data[feature_cols])
            except Exception as e:
                print(f"错误: 预测失败。请确保使用的是最新的模型文件。")
                raise e
        
        # 排序 (置信度从高到低)
        results = latest_data[['symbol', 'close', 'long_p', 'short_p', 'score']].sort_values('score', ascending=False)
        
        print("\n" + "="*70)
        print(f"PA/SMC 信号方向预测 ({tf_str}) - 美东时间 (ET)")
        print(f"分析时刻: {analysis_time}")
        print("-" * 70)
        print(f"{'代码':<8} | {'价格':<10} | {'做多置信度':<15} | {'做空置信度':<15}")
        print("-" * 70)
        
        for _, row in results.iterrows():
            print(f"{row['symbol']:<8} | {row['close']:<10.2f} | {row['long_p']:<15.2%} | {row['short_p']:<15.2%}")
            
        print("-" * 70)
        if len(results) > 1:
            top_row = results.iloc[0]
            direction = "Long 📈" if top_row['long_p'] > top_row['short_p'] else "Short 📉"
            top_conf = max(top_row['long_p'], top_row['short_p'])
            print(f"🚀 最强建议: {top_row['symbol']} [{direction}] (置信度: {top_conf:.1%})")
            
            # 显示置信度较高的方向
            high_long = results[results['long_p'] > 0.45]['symbol'].tolist()
            high_short = results[results['short_p'] > 0.45]['symbol'].tolist()
            if high_long: print(f"🐂 潜在做多: {', '.join(high_long)}")
            if high_short: print(f"🐻 潜在做空: {', '.join(high_short)}")
        else:
            row = results.iloc[0]
            if row['long_p'] > row['short_p']:
                status, icon = "多头 Setup", "🐂"
                conf = row['long_p']
            else:
                status, icon = "空头 Setup", "🐻"
                conf = row['short_p']
            
            if conf < 0.4: status, icon = "中性观察", "👀"
            print(f"{icon} {row['symbol']} 状态: {status} (置信度: {conf:.1%})")
        print("="*50)

    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_prediction()
