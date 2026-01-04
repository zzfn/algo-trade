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
    parser = argparse.ArgumentParser(description="Mag7 + 指数 排序策略回测工具")
    parser.add_argument("timeframe", nargs="?", default="1h", help="回测周期 (如 1d, 15m, 1h)")
    parser.add_argument("--days", type=int, default=365, help="回测天数 (默认 365 天)")
    parser.add_argument("--top_n", type=int, default=1, help="每天选取排名最高的前 N 个标的")
    parser.add_argument("--model", help="指定模型文件路径")
    
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

    if args.model:
        model_path = args.model
    else:
        model_path = f"output/mag7_{tf_str}_ranker.joblib"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}。")
        if not args.model:
            print(f"请先运行训练命令 (例如: make train-{tf_str})")
        return

    print(f"开始对 {len(symbols)} 个标的进行排序回测 (时间标准: 美东时间 ET)...")
    print(f"配置: 过去 {args.days} 天, Top {args.top_n}")
    
    try:
        # 1. 获取数据
        provider = DataProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days + 60)
        
        df_raw = provider.fetch_bars(symbols, timeframe, start_date, end_date)
        
        if df_raw.empty:
            print("错误: 未获取到数据。")
            return

        # 2. 特征工程
        builder = FeatureBuilder()
        # is_training=False 不产生 target_rank，只计算指标和 target_return
        df_features = builder.add_all_features(df_raw, is_training=False)
        
        # 过滤回测时段
        backtest_start = end_date - timedelta(days=args.days)
        df_test = df_features[df_features['timestamp'] >= backtest_start].copy()
        
        if df_test.empty:
            print("错误: 回测时段内无有效数据。")
            return

        # 3. 加载模型
        model = joblib.load(model_path)
        
        # 4. 特征列
        if "universal" in model_path.lower():
            feature_cols = [
                'return_1d', 'return_5d', 'ma_5_rel', 'ma_20_rel', 'ma_ratio', 'rsi', 
                'macd_rel', 'macd_signal_rel', 'macd_hist_rel', 'bb_upper_rel', 
                'bb_lower_rel', 'bb_width', 'volume_change', 'volume_ma_5', 
                'volume_ratio', 'volatility_20d', 'body_size_rel', 'candle_range_rel', 
                'upper_wick_rel', 'lower_wick_rel', 'wick_ratio', 'is_pin_bar', 
                'is_engulfing', 'swing_high', 'swing_low', 'bos_up', 'bos_down', 
                'fvg_up', 'fvg_down', 'fvg_size_rel', 'displacement', 'ob_bullish', 'ob_bearish'
            ]
        else:
            feature_cols = [
                'return_1d', 'return_5d', 'ma_5', 'ma_20', 
                'ma_ratio', 'rsi', 'volatility_20d',
                'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'volume_ratio', 'volume_change',
                'wick_ratio', 'is_pin_bar', 'is_engulfing',
                'fvg_up', 'fvg_down', 'displacement'
            ]
        
        # 5. 执行预测 (获取得分)
        df_test['score'] = model.predict(df_test[feature_cols])
        
        # 6. 核心逻辑：每天/每个周期选出 Top N
        # 我们需要计算每个 symbol 的 strategy_return
        # strategy_return = 如果该 symbol 被选中，则为它的下期收益，否则为 0
        
        # 6. 核心逻辑：每天/每个周期选出 Top N
        def pick_top_n(group):
            group = group.sort_values('score', ascending=False)
            group['is_selected'] = 0
            group.iloc[:args.top_n, group.columns.get_loc('is_selected')] = 1
            return group

        # 使用更稳健的循环方式避免 FutureWarning 和丢失 timestamp
        processed_ts = []
        for ts, group in df_test.groupby('timestamp'):
            processed = pick_top_n(group)
            processed['timestamp'] = ts
            processed_ts.append(processed)
        df_test = pd.concat(processed_ts).reset_index(drop=True)
        
        # 7. 计算每日策略总收益
        strategy_daily = df_test[df_test['is_selected'] == 1].groupby('timestamp')['target_return'].mean().fillna(0)
        
        # 8. 基准收益 (SPY 和 QQQ)
        spy_returns = df_test[df_test['symbol'] == 'SPY'].set_index('timestamp')['target_return']
        qqq_returns = df_test[df_test['symbol'] == 'QQQ'].set_index('timestamp')['target_return']
        
        # 9. 累积收益计算
        cum_strategy = (1 + strategy_daily).cumprod()
        cum_spy = (1 + spy_returns.fillna(0)).cumprod()
        cum_qqq = (1 + qqq_returns.fillna(0)).cumprod()
        
        # 指标计算
        total_strategy_ret = cum_strategy.iloc[-2] - 1 if len(cum_strategy) > 1 else 0
        total_spy_ret = cum_spy.iloc[-2] - 1 if len(cum_spy) > 1 else 0
        total_qqq_ret = cum_qqq.iloc[-2] - 1 if len(cum_qqq) > 1 else 0
        
        # 最大回撤
        roll_max = cum_strategy.cummax()
        dd = cum_strategy / roll_max - 1
        mdd = dd.min()

        print("\n" + "="*50)
        print(f"排序策略回测报告: {tf_str} (Top {args.top_n}) - ET")
        print(f"时间范围: {cum_strategy.index[0]} 至 {cum_strategy.index[-1]}")
        print(f"总周期数: {len(cum_strategy)}")
        print("-" * 50)
        print(f"策略累计收益 (Model): {total_strategy_ret:.2%}")
        print(f"SPY 累计收益 (基准): {total_spy_ret:.2%}")
        print(f"QQQ 累计收益 (基准): {total_qqq_ret:.2%}")
        print(f"最大回撤 (Max Drawdown): {mdd:.2%}")
        print("-" * 50)
        
        best_benchmark = max(total_spy_ret, total_qqq_ret)
        if total_strategy_ret > best_benchmark:
            print(f"结论: 🏆 [策略成功跑赢所有基准!]")
        elif total_strategy_ret > min(total_spy_ret, total_qqq_ret):
            print(f"结论: 📈 [策略表现尚可，优于部分基准]")
        else:
            print(f"结论: 📉 [策略表现逊于基准，需进一步优化]")
        
        print("="*50)

    except Exception as e:
        print(f"回测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_backtest()
