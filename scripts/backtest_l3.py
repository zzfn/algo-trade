# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from models.engine import StrategyEngine
from models.constants import L2_SYMBOLS, L3_LOOKBACK_DAYS, SIGNAL_THRESHOLD, get_feature_columns
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("l3_backtest_vbt_5min")

def run_l3_backtest_vbt(symbol, days=30, cash=10000.0):
    logger.info(f"🚀 开始 L3 (趋势确认) VectorBT 回测 (5min @ 1.0% Target): {symbol}, 回溯 {days} 天")
    
    engine = StrategyEngine()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    fetch_start = start_date - timedelta(days=L3_LOOKBACK_DAYS)
    
    # 1. 获取数据 (5m 频率 - 尝试降低噪音)
    logger.info(f"获取 {symbol} 5m K线数据...")
    df = engine.provider.fetch_bars(symbol, TimeFrame(5, TimeFrameUnit.Minute), fetch_start, end_date)
    
    if df.empty:
        logger.error("无数据")
        return
        
    # 添加特征
    df = engine.l2_builder.add_all_features(df, is_training=False)
    
    # 过滤测试区间
    df_test = df[df['timestamp'] >= start_date].copy()
    if df_test.empty:
        logger.error("测试区间无数据")
        return

    # 2. 预测 L3 信号
    logger.info("预测 L3 信号...")
    cols = get_feature_columns(df_test)
    probs = engine.l3_model.predict_proba(df_test[cols])
    df_test['long_p'] = probs[:, 1]
    df_test['short_p'] = probs[:, 2]
    
    # --- 计算动态风控参数 (关键更新) ---
    logger.info("计算 SMC 动态止盈止损参数...")
    from models.smc_rules import get_smc_risk_params
    
    # 初始化动态列
    df_test['sl_pct'] = np.nan
    df_test['tp_pct'] = np.nan
    
    # 只为超过阈值的信号计算风控参数 (节省计算开销)
    long_mask = df_test['long_p'] > SIGNAL_THRESHOLD
    short_mask = df_test['short_p'] > SIGNAL_THRESHOLD
    
    # 向量化处理由于 SMC 规则包含复杂逻辑，这里先用 apply 逐行处理信号点
    def apply_risk(row, direction):
        params = get_smc_risk_params(row, direction)
        return params['sl_pct'], params['tp_pct']

    if long_mask.any():
        res = df_test[long_mask].apply(lambda x: apply_risk(x, 'long'), axis=1)
        # VectorBT expects absolute percentages for stops
        df_test.loc[long_mask, 'sl_pct'] = [abs(r[0]) for r in res]
        df_test.loc[long_mask, 'tp_pct'] = [abs(r[1]) for r in res]
        
    if short_mask.any():
        res = df_test[short_mask].apply(lambda x: apply_risk(x, 'short'), axis=1)
        # 做空时，get_smc_risk_params 返回的 sl_pct 是正值，tp_pct 是负值
        # vbt 也期望绝对百分比
        df_test.loc[short_mask, 'sl_pct'] = [abs(r[0]) for r in res]
        df_test.loc[short_mask, 'tp_pct'] = [abs(r[1]) for r in res]

    # 准备 VectorBT 输入
    df_test.set_index('timestamp', inplace=True)
    close_prices = df_test['close']
    
    # 3. 生成信号和动态停止数组
    entries = df_test['long_p'] > SIGNAL_THRESHOLD
    short_entries = df_test['short_p'] > SIGNAL_THRESHOLD
    
    # 将 NaN 替换为 0 (0 表示不触发)，确保没有 negative values
    sl_stop = df_test['sl_pct'].fillna(0)
    tp_stop = df_test['tp_pct'].fillna(0)
    
    # 强制设置频率以便 vbt 计算年化
    df_test.index = pd.to_datetime(df_test.index)
    if df_test.index.freq is None:
        df_test = df_test.asfreq('5min').ffill() # 补全缺失数据以维持频率
        close_prices = df_test['close']
        entries = df_test['long_p'] > SIGNAL_THRESHOLD
        short_entries = df_test['short_p'] > SIGNAL_THRESHOLD
        sl_stop = df_test['sl_pct'].fillna(0)
        tp_stop = df_test['tp_pct'].fillna(0)

    logger.info(f"运行 VectorBT 组合回测 (动态风控, {df_test.index.freq})...")
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries,
        short_entries=short_entries,
        init_cash=cash,
        fees=0,      # 0.1% 手续费
        slippage=0,  # 0.1% 滑点
        sl_stop=sl_stop,   # 传入动态数组
        tp_stop=tp_stop,   # 传入动态数组
        freq='5min'        # 暂时保持 5min 测试
    )
    
    # 4. 输出结果
    # 5分钟频率，一年约 252 * 6.5 * 12 = 19656 个 bar
    # 强制传递 freq 给 stats 以便计算年化
    stats = portfolio.stats(settings=dict(ann_factor=19656))
    
    print("\n" + "="*60)
    print(f"📊 L3 趋势确认模型 (VectorBT) 回测结果: {symbol}")
    print("="*60)
    ann_return = stats.get('Annual Return [%]', 0)
    if ann_return == 0 and stats['Total Return [%]'] != 0:
        # 手动计算: (1 + total_return)^(year_fraction) - 1
        days_covered = (df_test.index[-1] - df_test.index[0]).days
        if days_covered > 0:
            ann_return = ((1 + stats['Total Return [%]']/100) ** (365.25 / days_covered) - 1) * 100
            
    print(f"总收益率:        {stats['Total Return [%]']:.2f}%")
    print(f"年化收益率:      {ann_return:.2f}%")
    print(f"夏普比率:        {stats.get('Sharpe Ratio', 0):.2f}")
    print(f"最大回撤:        {stats['Max Drawdown [%]']:.2f}%")
    print(f"最大回撤时长:    {stats.get('Max Drawdown Duration', 'N/A')}")
    print("-" * 30)
    print(f"总交易次数:      {int(stats['Total Trades'])}")
    print(f"胜率:            {stats.get('Win Rate [%]', 0):.2f}%")
    print(f"利润因子:        {stats.get('Profit Factor', np.nan):.2f}")
    print(f"期望盈亏 (Expectancy): {stats.get('Expectancy', np.nan):.4f}")
    
    # 修复 Avg Win / Avg Loss Ratio
    wl_ratio = stats.get('Avg Win / Avg Loss Ratio', 0)
    if not np.isfinite(wl_ratio) or wl_ratio == 0:
        trades_rec = portfolio.trades.records_readable
        if not trades_rec.empty:
            ret_col = 'Return' if 'Return' in trades_rec.columns else 'Return [%]'
            avg_win = trades_rec[trades_rec['PnL'] > 0][ret_col].mean()
            avg_loss = abs(trades_rec[trades_rec['PnL'] < 0][ret_col].mean())
            wl_ratio = avg_win / avg_loss if avg_loss != 0 else np.nan
    print(f"平均盈利/平均亏损: {wl_ratio:.2f}")
    
    # 5. 多空细节
    trades = portfolio.trades.records_readable
    if not trades.empty:
        long_trades = trades[trades['Direction'] == 'Long']
        short_trades = trades[trades['Direction'] == 'Short']
        
        print("-" * 30)
        print(f"做多交易: {len(long_trades)} 次 | 胜率: {(long_trades['PnL'] > 0).mean()*100:.1f}%")
        print(f"做空交易: {len(short_trades)} 次 | 胜率: {(short_trades['PnL'] > 0).mean()*100:.1f}%")
        
        # 最好/最差交易
        print("-" * 30)
        # 调试列名: print(trades.columns)
        return_col = 'Return' if 'Return' in trades.columns else 'Return [%]'
        best_trade = trades.loc[trades['PnL'].idxmax()]
        worst_trade = trades.loc[trades['PnL'].idxmin()]
        print(f"最大盈利: {best_trade[return_col] * 100:.2f}% ({best_trade['Exit Timestamp']})")
        print(f"最大亏损: {worst_trade[return_col] * 100:.2f}% ({worst_trade['Exit Timestamp']})")
        
    print(f"="*60)
    
    # 保存报告
    report_path = f"reports/backtest_l3_{symbol}.html"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    portfolio.plot().write_html(report_path)
    logger.info(f"📈 VBT 报告已保存至: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--cash", type=float, default=10000.0)
    args = parser.parse_args()
    
    run_l3_backtest_vbt(args.symbol, args.days, args.cash)
