
import argparse
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from models.engine import StrategyEngine
from models.constants import L2_SYMBOLS, L3_LOOKBACK_DAYS, SIGNAL_THRESHOLD, get_feature_columns
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("l3_backtest")

def run_l3_backtest(symbol, days=30):
    logger.info(f"🚀 开始 L3 (择时信号) 回测: {symbol}, 回溯 {days} 天")
    
    engine = StrategyEngine()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    fetch_start = start_date - timedelta(days=L3_LOOKBACK_DAYS)
    
    # 1. 获取 15m 数据
    logger.info(f"获取 {symbol} 15m K线数据...")
    df = engine.provider.fetch_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), fetch_start, end_date)
    
    if df.empty:
        logger.error("无数据")
        return
        
    df = engine.l2_builder.add_all_features(df, is_training=False)
    
    df_test = df[df['timestamp'] >= start_date].copy()
    if df_test.empty:
        logger.error("测试区间无数据")
        return

    # 2. 预测信号
    logger.info("预测 L3 信号...")
    cols = get_feature_columns(df_test)
    probs = engine.l3_model.predict_proba(df_test[cols])
    df_test['long_p'] = probs[:, 1]
    df_test['short_p'] = probs[:, 2]
    
    # 3. 模拟交易
    # 简单策略:
    # long_p > THRESHOLD -> Open Long
    # short_p > THRESHOLD -> Open Short
    # 持仓直到反向信号或固定止盈止损 (这里用 2% SL/TP 简化测试信号质量)
    
    balance = 10000.0
    position = 0 # size
    entry_price = 0
    df_test = df_test.reset_index(drop=True)
    trades = []
    
    # 简化回测：逐行遍历
    state = 'flat' # flat, long, short
    
    for i, row in df_test.iterrows():
        price = row['close']
        ts = row['timestamp']
        
        # 检查平仓
        if state == 'long':
            # Stop Loss / Take Profit (Fixed 2%)
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct < -0.02 or pnl_pct > 0.04 or row['short_p'] > SIGNAL_THRESHOLD:
                # Close Long
                pnl = (price - entry_price) * position
                balance += price * position
                trades.append({'time': ts, 'type': 'close_long', 'price': price, 'pnl': pnl, 'reason': 'signal' if row['short_p'] > SIGNAL_THRESHOLD else 'limit'})
                state = 'flat'
                position = 0
                
        elif state == 'short':
             pnl_pct = (entry_price - price) / entry_price
             if pnl_pct < -0.02 or pnl_pct > 0.04 or row['long_p'] > SIGNAL_THRESHOLD:
                # Close Short
                # Buy back
                cost = price * position # cash needed to buy back
                pnl = (entry_price - price) * position # profit
                # balance logic for short is tricky in simple simulation
                # let's just add pnl to balance
                balance += pnl 
                trades.append({'time': ts, 'type': 'close_short', 'price': price, 'pnl': pnl, 'reason': 'signal' if row['long_p'] > SIGNAL_THRESHOLD else 'limit'})
                state = 'flat'
                position = 0

        # 检查开仓
        if state == 'flat':
            if row['long_p'] > SIGNAL_THRESHOLD:
                # Open Long
                entry_price = price
                position = int(balance / price) # Full port
                balance -= position * entry_price
                state = 'long'
                trades.append({'time': ts, 'type': 'open_long', 'price': price, 'prob': row['long_p']})
            elif row['short_p'] > SIGNAL_THRESHOLD:
                # Open Short
                entry_price = price
                position = int(balance / price)
                # assume we have margin
                state = 'short'
                trades.append({'time': ts, 'type': 'open_short', 'price': price, 'prob': row['short_p']})

    # 强制平仓
    if state != 'flat':
        curr_price = df_test.iloc[-1]['close']
        if state == 'long':
            pnl = (curr_price - entry_price) * position
            balance += curr_price * position
        else:
            pnl = (entry_price - curr_price) * position
            balance += pnl
        # Add back initial cash for short? No simplified above.
        # simpler: Final Equity = Balance + Market Value of Postions
    
    final_equity = balance
    if state == 'long': # balance is low, holding stock
        final_equity = balance + df_test.iloc[-1]['close'] * position
    elif state == 'short': # balance is high (short proceeds), need to buy back
        pass # Simplified above logic PnL added directly to balance

    # Recalculate correctly
    # Let's use simple cumulative PnL
    total_pnl = sum(t['pnl'] for t in trades if 'pnl' in t)
    
    print("\n" + "="*60)
    print(f"📊 L3 执行信号模型回测结果 ({symbol})")
    print("="*60)
    print(f"交易次数: {len([t for t in trades if 'close' in t['type']])}")
    print(f"总 PnL: ${total_pnl:.2f}")
    
    wins = [t for t in trades if 'pnl' in t and t['pnl'] > 0]
    losses = [t for t in trades if 'pnl' in t and t['pnl'] <= 0]
    win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0
    
    print(f"胜率: {win_rate:.1%}")
    if trades:
        print("\n最近 5 笔交易:")
        for t in trades[-5:]:
            print(t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    
    run_l3_backtest(args.symbol, args.days)
