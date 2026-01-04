import os
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from data.provider import DataProvider
from features.technical import FeatureBuilder

# 加载环境变量
load_dotenv()

def run_backtest():
    parser = argparse.ArgumentParser(description="Mag7 + 指数 排序策略回测工具")
    parser.add_argument("timeframe", nargs="?", default="1h", help="回测周期 (如 1d, 15m, 1h)")
    parser.add_argument("--days", type=int, default=365, help="回测天数 (默认 365 天)")
    parser.add_argument("--top_n", type=int, default=1, help="每天选取排名最高的前 N 个标的")
    parser.add_argument("--model", help="指定模型文件路径")
    parser.add_argument("--details", action="store_true", help="打印详细交易记录")
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
        model_path = f"models/artifacts/mag7_{tf_str}_ranker.joblib"
    
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
        
        # 4. 定义特征排除列表 (与训练代码保持一致)
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                        'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                        'max_future_return', 'target_signal', 'local_high', 'local_low']
        
        feature_cols = [c for c in df_test.columns if c not in exclude_cols]
        print(f"输入特征维度: {len(feature_cols)}")
        
        # 5. 执行预测 (获取得分)
        df_test['score'] = model.predict(df_test[feature_cols])
        
        # 6. 核心逻辑：多空策略 - 同时选择得分最高(做多)和最低(做空)的标的
        def pick_long_short(group):
            group = group.sort_values('score', ascending=False)
            group['position'] = 'NONE'
            # 选择 top_n 做多
            if len(group) >= args.top_n:
                group.iloc[:args.top_n, group.columns.get_loc('position')] = 'LONG'
            # 选择 bottom_n 做空
            if len(group) >= args.top_n * 2:
                group.iloc[-args.top_n:, group.columns.get_loc('position')] = 'SHORT'
            return group

        # 使用更稳健的循环方式避免 FutureWarning 和丢失 timestamp
        processed_ts = []
        for ts, group in df_test.groupby('timestamp'):
            processed = pick_long_short(group)
            processed['timestamp'] = ts
            processed_ts.append(processed)
        df_test = pd.concat(processed_ts).reset_index(drop=True)
        
        # 7. 计算多空策略收益
        # 做多收益 = 持有多头标的的平均收益
        long_daily = df_test[df_test['position'] == 'LONG'].groupby('timestamp')['target_return'].mean()
        
        # 做空收益 = 持有空头标的的平均收益(取反,因为做空时价格下跌=盈利)
        # 例如: 标的跌 -2% → 做空盈利 +2%
        short_daily = -df_test[df_test['position'] == 'SHORT'].groupby('timestamp')['target_return'].mean()
        
        # 多空对冲策略收益 = (做多收益 + 做空收益) / 2
        # 注意: 这里改为相加,因为 short_daily 已经取反
        strategy_daily = ((long_daily.fillna(0) + short_daily.fillna(0)) / 2)
        
        # 调试输出
        print("\n[DEBUG] 收益计算详情:")
        for ts in strategy_daily.index[:5]:  # 只显示前5个
            long_ret = long_daily.get(ts, 0)
            short_ret = short_daily.get(ts, 0)
            strat_ret = strategy_daily.get(ts, 0)
            print(f"  {ts}: LONG={long_ret:+.4f} | SHORT={short_ret:+.4f} | STRATEGY={strat_ret:+.4f}")
        print()
        
        # 8. 打印交易细节 (如果启用)
        if args.details:
            print("\n" + "-"*110)
            print(f"{'时间 (ET)':<20} | {'方向':<6} | {'代码':<8} | {'收盘价':<10} | {'预测分':<10} | {'标的涨跌':<10} | {'策略收益':<10}")
            print("-"*110)
            # 获取所有被选中的行
            selected_trades = df_test[df_test['position'] != 'NONE'].sort_values('timestamp')
            for _, row in selected_trades.iterrows():
                direction_icon = "📈" if row['position'] == 'LONG' else "📉"
                # 计算策略收益: 做多=标的收益, 做空=标的收益取反
                strategy_return = row['target_return'] if row['position'] == 'LONG' else -row['target_return']
                print(f"{str(row['timestamp']):<20} | {direction_icon} {row['position']:<4} | {row['symbol']:<8} | {row['close']:<10.2f} | {row['score']:<10.4f} | {row['target_return']:+10.2%} | {strategy_return:+10.2%}")
            print("-"*110 + "\n")

        # 9. 基准收益 (SPY 和 QQQ)
        spy_returns = df_test[df_test['symbol'] == 'SPY'].set_index('timestamp')['target_return']
        qqq_returns = df_test[df_test['symbol'] == 'QQQ'].set_index('timestamp')['target_return']
        
        # 9. 累积收益计算
        cum_strategy = (1 + strategy_daily).cumprod()
        cum_spy = (1 + spy_returns.fillna(0)).cumprod()
        cum_qqq = (1 + qqq_returns.fillna(0)).cumprod()
        
        # 指标计算
        # 修复: 使用 iloc[-1] 获取最后的累积收益,而不是 iloc[-2]
        total_strategy_ret = cum_strategy.iloc[-1] - 1 if len(cum_strategy) > 0 else 0
        total_spy_ret = cum_spy.iloc[-1] - 1 if len(cum_spy) > 0 else 0
        total_qqq_ret = cum_qqq.iloc[-1] - 1 if len(cum_qqq) > 0 else 0
        
        # 调试累积收益
        print(f"[DEBUG] 累积收益序列:")
        print(f"  strategy_daily: {strategy_daily.tolist()}")
        print(f"  cum_strategy: {cum_strategy.tolist()}")
        print(f"  最终累积收益: {total_strategy_ret:.4%}")
        print()
        
        # 最大回撤
        roll_max = cum_strategy.cummax()
        dd = cum_strategy / roll_max - 1
        mdd = dd.min()

        print("\n" + "="*50)
        print(f"多空策略回测报告: {tf_str} (Long {args.top_n} + Short {args.top_n}) - ET")
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
