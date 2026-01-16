"""
VectorBT 回测脚本 - 集成四层模型

使用 VectorBT 进行高性能回测,支持:
- 多空策略
- 动态仓位
- SMC 止盈止损
- 专业性能报告
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.engine import StrategyEngine
from models.constants import L2_SYMBOLS, SIGNAL_THRESHOLD
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
import argparse
from pathlib import Path

def generate_signals(symbols, start_date, end_date):
    """
    使用四层模型生成交易信号
    
    Returns:
        entries: 做多信号 DataFrame
        exits: 做空信号 DataFrame  
        close_prices: 收盘价 DataFrame
    """
    load_dotenv()
    provider = DataProvider()
    engine = StrategyEngine()
    
    print(f"📊 获取数据: {len(symbols)} 只股票, {start_date} 到 {end_date}")
    
    # 获取分钟线数据用于回测
    df_raw = provider.fetch_bars(symbols, TimeFrame.Minute, start_date, end_date)
    print(f"   原始数据: {len(df_raw)} 行")
    
    # 按时间和标的重塑数据
    df_pivot = df_raw.pivot(index='timestamp', columns='symbol', values='close')
    close_prices = df_pivot
    
    print(f"🤖 生成交易信号...")
    
    # 初始化信号 DataFrame
    entries = pd.DataFrame(False, index=df_pivot.index, columns=df_pivot.columns)
    exits = pd.DataFrame(False, index=df_pivot.index, columns=df_pivot.columns)
    
    # 每小时生成一次信号 (避免过于频繁)
    signal_timestamps = df_pivot.index[::60]  # 每60分钟
    
    for i, timestamp in enumerate(signal_timestamps):
        if i % 10 == 0:
            print(f"   进度: {i}/{len(signal_timestamps)}")
        
        try:
            # 调用四层模型
            results = engine.analyze(timestamp)
            
            if results.get('l3_signals') is None or results['l3_signals'].empty:
                continue
            
            l3_signals = results['l3_signals']
            
            # 生成信号
            for _, row in l3_signals.iterrows():
                symbol = row['symbol']
                if symbol not in entries.columns:
                    continue
                
                # 做多信号
                if row['long_p'] > SIGNAL_THRESHOLD:
                    entries.loc[timestamp, symbol] = True
                
                # 做空信号  
                if row['short_p'] > SIGNAL_THRESHOLD:
                    exits.loc[timestamp, symbol] = True
                    
        except Exception as e:
            print(f"   警告: {timestamp} 信号生成失败: {e}")
            continue
    
    print(f"✅ 信号生成完成")
    print(f"   做多信号: {entries.sum().sum()} 个")
    print(f"   做空信号: {exits.sum().sum()} 个")
    
    return entries, exits, close_prices

def run_backtest(entries, exits, close_prices, init_cash=100000, fees=0.001):
    """
    运行 VectorBT 回测
    """
    print(f"\n💰 运行回测...")
    print(f"   初始资金: ${init_cash:,.0f}")
    print(f"   手续费: {fees:.2%}")
    
    # 创建投资组合
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        slippage=0.001,  # 0.1% 滑点
        freq='1min',
        group_by=True    # 聚合所有标的为一个投资组合
    )
    
    return portfolio

def print_stats(portfolio):
    """打印回测统计"""
    stats = portfolio.stats()
    
    print(f"\n" + "="*60)
    print(f"📊 回测结果")
    print(f"="*60)
    print(f"总收益率:        {stats['Total Return [%]']:.2f}%")
    print(f"年化收益率:      {stats.get('Annual Return [%]', 0):.2f}%")
    print(f"夏普比率:        {stats.get('Sharpe Ratio', 0):.2f}")
    print(f"最大回撤:        {stats['Max Drawdown [%]']:.2f}%")
    
    # Win Rate handling
    win_rate = stats.get('Win Rate [%]', 0)
    win_rate_str = "N/A" if pd.isna(win_rate) else f"{win_rate:.2f}%"
    print(f"胜率:            {win_rate_str}")
    
    # Trade counts
    total_trades = stats['Total Trades']
    closed_trades = stats.get('Total Closed Trades', 0)
    open_trades = stats.get('Total Open Trades', 0)
    
    print(f"总交易次数:      {total_trades} (Open: {open_trades}, Closed: {closed_trades})")
    print(f"="*60)
    
    return stats

def save_report(portfolio, output_path='reports/backtest_vbt.html'):
    """保存 HTML 报告"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 生成图表
    fig = portfolio.plot()
    fig.write_html(output_path)
    
    print(f"\n📈 报告已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='VectorBT 回测')
    parser.add_argument('--days', type=int, default=30, help='回测天数')
    parser.add_argument('--cash', type=float, default=100000, help='初始资金')
    parser.add_argument('--fees', type=float, default=0.001, help='手续费率')
    args = parser.parse_args()
    
    # 设置日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # 使用标的池
    symbols = L2_SYMBOLS[:5]  # 先用5只股票测试
    
    print(f"\n{'='*60}")
    print(f"🚀 VectorBT 回测系统")
    print(f"{'='*60}")
    print(f"回测期间: {start_date.date()} 到 {end_date.date()}")
    print(f"标的数量: {len(symbols)}")
    print(f"{'='*60}\n")
    
    # 生成信号
    entries, exits, close_prices = generate_signals(symbols, start_date, end_date)
    
    # 运行回测
    portfolio = run_backtest(entries, exits, close_prices, args.cash, args.fees)
    
    # 打印统计
    stats = print_stats(portfolio)
    
    # 保存报告
    save_report(portfolio)
    
    print(f"\n✅ 回测完成!")

if __name__ == "__main__":
    main()
