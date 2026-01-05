
import argparse
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from models.engine import StrategyEngine
from models.constants import L2_SYMBOLS, get_feature_columns, TOP_N_TRADES
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("l4_backtest")

def run_l4_backtest(days=60):
    logger.info(f"🚀 开始 L4 (风控与仓位管理) 回测, 回溯 {days} 天")
    
    engine = StrategyEngine()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 1. 获取数据 (选取几个主要标的进行测试)
    test_symbols = ['NVDA', 'TSLA', 'AAPL', 'AMD']
    logger.info(f"测试标的: {test_symbols}")
    
    # 获取 5m 数据以更精确模拟盘中价格行为 (SMC 需要 High/Low)
    fetch_start = start_date - timedelta(days=10) # buffer
    
    # 我们不仅需要测试 仓位管理 (Allocation)，还需要测试 SMC 风控 (SL/TP)
    # 对比策略:
    # A. 基准: 固定仓位 (10%), 固定止盈止损 (TP 5%, SL 2%)
    # B. L4策略: 动态仓位 (基于预测收益), SMC 动态止盈止损
    
    results = []
    
    for sym in test_symbols:
        logger.info(f"回测 {sym} ...")
        df = engine.provider.fetch_bars(sym, TimeFrame(5, TimeFrameUnit.Minute), fetch_start, end_date)
        if df.empty:
            continue
            
        # 构造日线级别的特征用于 L4 预测 (通常 L4 基于日线或小时线特征预测收益)
        # 这里为了简化，我们每隔 4 小时做一次决策
        df = engine.l2_builder.add_all_features(df, is_training=False)
        df = df[df['timestamp'] >= start_date].reset_index(drop=True)
        
        # 模拟
        balance_a = 10000.0 # Fixed
        balance_b = 10000.0 # L4
        
        pos_a = None
        pos_b = None
        
        # 为了加速，简化循环：每 4 小时尝试开仓
        # 实际上应该用 event-driven，这里用简单的时间步进
        
        # 重新采样到 1h 进行决策，但用 5m 数据进行撮合？
        # 简化: 直接遍历 5m 数据，每 12 根 bar (1小时) 检查一次开仓信号
        # 假设总是做多 (为了测试风控能力，忽略择时)
        
        for i in range(0, len(df), 12):
            if i + 12 >= len(df): break
            
            bar = df.iloc[i]
            ts = bar['timestamp']
            price = bar['close']
            
            # --- 策略 A: 固定 ---
            if pos_a is None:
                # 开仓
                size = int((10000 * 0.1) / price) # 假设总资金恒定 10k 计算仓位
                tp = price * 1.05
                sl = price * 0.98
                pos_a = {'entry': price, 'size': size, 'tp': tp, 'sl': sl, 'ts': ts}
            
            # --- 策略 B: L4 动态 ---
            if pos_b is None:
                # 预测收益
                # 构造单行 DataFrame
                cols = get_feature_columns(df)
                l2_df = pd.DataFrame([bar])
                
                # 1. 动态仓位
                alloc = engine.get_allocation(sym, l2_df)
                target_val = 10000 * alloc
                size_b = int(target_val / price)
                
                # 2. SMC 风控
                risk = engine.get_risk_params(sym, 'long', l2_df)
                if risk:
                    tp_b = risk['take_profit']
                    sl_b = risk['stop_loss']
                    pos_b = {'entry': price, 'size': size_b, 'tp': tp_b, 'sl': sl_b, 'ts': ts}
            
            # --- 撮合 (检查未来 12 根 5m K线) ---
            chunk = df.iloc[i+1 : i+13]
            
            # Check A
            if pos_a:
                done = False
                for _, row in chunk.iterrows():
                    if row['low'] <= pos_a['sl']:
                        # Stop Loss
                        pnl = (pos_a['sl'] - pos_a['entry']) * pos_a['size']
                        balance_a += pnl
                        pos_a = None
                        done = True
                        break
                    elif row['high'] >= pos_a['tp']:
                        # Take Profit
                        pnl = (pos_a['tp'] - pos_a['entry']) * pos_a['size']
                        balance_a += pnl
                        pos_a = None
                        done = True
                        break
                # Period end close (Time exit? No, hold until SL/TP for this test)
                # But to avoid holding forever in this loop, let's say we refresh logic?
                # For simplicity, keep holding if not hit.
            
            # Check B
            if pos_b:
                done = False
                for _, row in chunk.iterrows():
                    if row['low'] <= pos_b['sl']:
                        pnl = (pos_b['sl'] - pos_b['entry']) * pos_b['size']
                        balance_b += pnl
                        pos_b = None
                        done = True
                        break
                    elif row['high'] >= pos_b['tp']:
                        pnl = (pos_b['tp'] - pos_b['entry']) * pos_b['size']
                        balance_b += pnl
                        pos_b = None
                        done = True
                        break
        
        # End of symbol loop
        results.append({
            'symbol': sym,
            'fixed_pnl': balance_a - 10000,
            'l4_pnl': balance_b - 10000
        })

    print("\n" + "="*60)
    print("📊 L4 风控模型 (SMC + Alloc) 对比回测")
    print("="*60)
    print(f"回测区间: {start_date.date()} ~ {end_date.date()}")
    print(f"对比策略: 固定 10%仓位+2%/5%止损盈 VS L4动态仓位+SMC止损盈")
    
    print("-" * 60)
    print(f"{'标的':<6} | {'固定策略 PnL':<15} | {'L4 策略 PnL':<15} | {'差异':<10}")
    print("-" * 60)
    
    total_fixed = 0
    total_l4 = 0
    for res in results:
        diff = res['l4_pnl'] - res['fixed_pnl']
        total_fixed += res['fixed_pnl']
        total_l4 += res['l4_pnl']
        icon = "✅" if diff > 0 else "❌"
        print(f"{res['symbol']:<6} | ${res['fixed_pnl']:<14.2f} | ${res['l4_pnl']:<14.2f} | {icon} {diff:+.2f}")
        
    print("-" * 60)
    print(f"总计   | ${total_fixed:<14.2f} | ${total_l4:<14.2f} | {total_l4 - total_fixed:+.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()
    
    run_l4_backtest(args.days)
