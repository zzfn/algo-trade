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
from models.constants import get_feature_columns

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
        feature_cols = get_feature_columns(df_test)
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
        # 使用 align 和 fill_value=0 确保如果某个时刻只有单边信号也能计算
        strategy_daily = long_daily.add(short_daily, fill_value=0) / 2
        
        
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
        # 修复：删除最后一个 NaN (因为最后一个时间点没有未来收益数据)
        strategy_daily = strategy_daily.dropna()
        spy_returns = spy_returns.dropna()
        qqq_returns = qqq_returns.dropna()

        cum_strategy = (1 + strategy_daily).cumprod()
        cum_spy = (1 + spy_returns).cumprod()
        cum_qqq = (1 + qqq_returns).cumprod()
        
        # 指标计算
        # 修复: 使用 iloc[-1] 获取最后的累积收益,而不是 iloc[-2]
        total_strategy_ret = cum_strategy.iloc[-1] - 1 if len(cum_strategy) > 0 else 0
        total_spy_ret = cum_spy.iloc[-1] - 1 if len(cum_spy) > 0 else 0
        total_qqq_ret = cum_qqq.iloc[-1] - 1 if len(cum_qqq) > 0 else 0
        
        
        # 最大回撤
        roll_max = cum_strategy.cummax()
        dd = cum_strategy / roll_max - 1
        mdd = dd.min()
        
        # === 新增指标 ===
        # 1. 胜率 (Win Rate)
        wins = (strategy_daily > 0).sum()
        losses = (strategy_daily < 0).sum()
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # 2. 盈亏比 (Profit Factor)
        gross_profit = strategy_daily[strategy_daily > 0].sum()
        gross_loss = abs(strategy_daily[strategy_daily < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 3. 夏普比率 (Sharpe Ratio) - 假设无风险利率为 0
        daily_mean = strategy_daily.mean()
        daily_std = strategy_daily.std()
        # 根据周期调整年化因子
        if tf_str == '1d':
            annualization_factor = 252
        elif tf_str == '1h':
            annualization_factor = 252 * 6.5  # 每天 6.5 个交易小时
        elif tf_str.endswith('m'):
            mins = int(tf_str.replace('m', ''))
            annualization_factor = 252 * 6.5 * (60 / mins)
        else:
            annualization_factor = 252
        sharpe_ratio = (daily_mean / daily_std * np.sqrt(annualization_factor)) if daily_std > 0 else 0
        
        # 4. 年化收益率
        trading_days = len(strategy_daily)
        if tf_str == '1d':
            years = trading_days / 252
        elif tf_str == '1h':
            years = trading_days / (252 * 6.5)
        else:
            mins = int(tf_str.replace('m', '')) if tf_str.endswith('m') else 60
            years = trading_days / (252 * 6.5 * (60 / mins))
        annual_return = (1 + total_strategy_ret) ** (1 / years) - 1 if years > 0 else 0
        
        # 5. 平均每笔收益
        avg_return = strategy_daily.mean()
        
        # 6. 最大连续亏损次数
        losing_streak = 0
        max_losing_streak = 0
        for r in strategy_daily:
            if r < 0:
                losing_streak += 1
                max_losing_streak = max(max_losing_streak, losing_streak)
            else:
                losing_streak = 0
        
        # 7. 卡玛比率 (Calmar Ratio) = 年化收益 / 最大回撤
        calmar_ratio = annual_return / abs(mdd) if mdd != 0 else float('inf')

        print("\n" + "="*60)
        print(f"多空策略回测报告: {tf_str} (Long {args.top_n} + Short {args.top_n}) - ET")
        print(f"时间范围: {cum_strategy.index[0]} 至 {cum_strategy.index[-1]}")
        print(f"总周期数: {len(cum_strategy)}")
        print("="*60)
        
        print("\n📊 收益指标:")
        print("-" * 60)
        print(f"  策略累计收益: {total_strategy_ret:>10.2%}    SPY: {total_spy_ret:>8.2%}    QQQ: {total_qqq_ret:>8.2%}")
        print(f"  年化收益率:   {annual_return:>10.2%}")
        print(f"  平均每周期:   {avg_return:>10.4%}")
        
        print("\n📉 风险指标:")
        print("-" * 60)
        print(f"  最大回撤:     {mdd:>10.2%}")
        print(f"  波动率 (std): {daily_std:>10.4%}")
        print(f"  最大连续亏损: {max_losing_streak:>10} 次")
        
        print("\n⚖️ 风险调整指标:")
        print("-" * 60)
        print(f"  夏普比率:     {sharpe_ratio:>10.2f}")
        print(f"  卡玛比率:     {calmar_ratio:>10.2f}")
        print(f"  盈亏比:       {profit_factor:>10.2f}")
        
        print("\n🎯 交易统计:")
        print("-" * 60)
        print(f"  总交易周期:   {total_trades:>10}")
        print(f"  盈利周期:     {wins:>10} ({win_rate:.1%})")
        print(f"  亏损周期:     {losses:>10} ({1-win_rate:.1%})")
        
        print("\n" + "="*60)
        best_benchmark = max(total_spy_ret, total_qqq_ret)
        if total_strategy_ret > best_benchmark:
            print(f"结论: 🏆 [策略成功跑赢所有基准!]")
        elif total_strategy_ret > min(total_spy_ret, total_qqq_ret):
            print(f"结论: 📈 [策略表现尚可，优于部分基准]")
        else:
            print(f"结论: 📉 [策略表现逊于基准，需进一步优化]")
        
        # 策略改进建议
        print("\n💡 调优建议:")
        if win_rate < 0.5:
            print("  - 胜率较低，考虑提高信号阈值或增加过滤条件")
        if profit_factor < 1.5:
            print("  - 盈亏比偏低，考虑优化止盈止损参数")
        if sharpe_ratio < 1.0:
            print("  - 夏普比率不足，收益相对风险偏低")
        if abs(mdd) > 0.1:
            print("  - 回撤较大，考虑增加风控或降低仓位")
        if max_losing_streak > 5:
            print("  - 连续亏损过多，可能存在趋势判断问题")
        if win_rate >= 0.5 and profit_factor >= 1.5 and sharpe_ratio >= 1.0:
            print("  - ✅ 各项指标健康，可考虑扩大回测时间验证稳定性")
        
        print("="*60)

    except Exception as e:
        print(f"回测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_backtest()
