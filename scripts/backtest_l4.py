
import argparse
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from strategies.engine import StrategyEngine
from config.settings import L2_SYMBOLS, get_feature_columns, TOP_N_TRADES, get_allocation_by_return
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
    
    results = []
    
    for sym in test_symbols:
        logger.info(f"回测 {sym} ...")
        df = engine.provider.fetch_bars(sym, TimeFrame(5, TimeFrameUnit.Minute), fetch_start, end_date)
        if df.empty:
            continue
            
        # 构造日线级别的特征
        df = engine.l2_builder.add_all_features(df, is_training=False)
        df = df[df['timestamp'] >= start_date].reset_index(drop=True)
        
        # 模拟
        balance_a = 10000.0 # Fixed
        balance_b = 10000.0 # L4
        
        pos_a = None
        pos_b = None
        
        # 统计 L4 交易详情
        l4_trade_count = 0
        l4_avg_alloc = 0.0
        
        # 简化循环：每 12 根 bar (1小时) 检查一次开仓信号
        # 假设总是做多 (为了测试风控能力，忽略择时)
        
        for i in range(0, len(df), 12):
            if i + 12 >= len(df): break
            
            bar = df.iloc[i]
            ts = bar['timestamp']
            price = bar['close']
            
            # --- 策略 A: 固定 10% 仓位, 5% TP, 2% SL ---
            if pos_a is None:
                size = int((10000 * 0.1) / price)
                tp = price * 1.05
                sl = price * 0.98
                pos_a = {'entry': price, 'size': size, 'tp': tp, 'sl': sl, 'ts': ts}
            
            # --- 策略 B: L4 动态 ---
            if pos_b is None:
                # 构造单行 DataFrame 用于预测
                l2_df = pd.DataFrame([bar])
                
                # 1. 动态仓位 (Debug)
                pred_ret = engine.predict_return(sym, l2_df)
                alloc = get_allocation_by_return(pred_ret)
                
                target_val = 10000 * alloc
                size_b = int(target_val / price)
                
                if size_b > 0:
                    # 2. SMC 风控
                    risk = engine.get_risk_params(sym, 'long', l2_df)
                    if risk:
                        tp_b = risk['take_profit']
                        sl_b = risk['stop_loss']
                        pos_b = {'entry': price, 'size': size_b, 'tp': tp_b, 'sl': sl_b, 'ts': ts}
                        
                        l4_trade_count += 1
                        l4_avg_alloc += alloc
                        
                        # Debug Log (抽样打印)
                        if l4_trade_count % 20 == 0:
                            sl_dist = (risk['stop_loss'] / price) - 1
                            tp_dist = (risk['take_profit'] / price) - 1
                            logger.info(f"[{ts}] L4 Trade: Pred={pred_ret:.4%}, Alloc={alloc:.2%}, Size={size_b}, SL={sl_dist:.2%}, TP={tp_dist:.2%}")
            
            # --- 撮合 (检查未来 12 根 5m K线) ---
            chunk = df.iloc[i+1 : i+13]
            
            # Check A
            if pos_a:
                for _, row in chunk.iterrows():
                    if row['low'] <= pos_a['sl']:
                        pnl = (pos_a['sl'] - pos_a['entry']) * pos_a['size']
                        balance_a += pnl
                        pos_a = None
                        break
                    elif row['high'] >= pos_a['tp']:
                        pnl = (pos_a['tp'] - pos_a['entry']) * pos_a['size']
                        balance_a += pnl
                        pos_a = None
                        break
            
            # Check B
            if pos_b:
                for _, row in chunk.iterrows():
                    if row['low'] <= pos_b['sl']:
                        pnl = (pos_b['sl'] - pos_b['entry']) * pos_b['size']
                        balance_b += pnl
                        pos_b = None
                        break
                    elif row['high'] >= pos_b['tp']:
                        pnl = (pos_b['tp'] - pos_b['entry']) * pos_b['size']
                        balance_b += pnl
                        pos_b = None
                        break
        
        # End of symbol loop
        l4_avg_alloc = l4_avg_alloc / l4_trade_count if l4_trade_count > 0 else 0
        logger.info(f"{sym} Summary: L4 Trades={l4_trade_count}, AvgAlloc={l4_avg_alloc:.2%}, FixedPnL=${balance_a-10000:.2f}, L4PnL=${balance_b-10000:.2f}")
        
        results.append({
            'symbol': sym,
            'fixed_pnl': balance_a - 10000,
            'l4_pnl': balance_b - 10000,
            'l4_trades': l4_trade_count,
            'l4_avg_alloc': l4_avg_alloc
        })

    print("\n" + "="*60)
    print("📊 L4 风控模型 (SMC + Alloc) 对比回测")
    print("="*60)
    print(f"回测区间: {start_date.date()} ~ {end_date.date()}")
    print(f"对比策略: 固定 10%仓位+2%/5%止损盈 VS L4动态仓位+SMC止损盈")
    
    print("-" * 80)
    print(f"{'标的':<6} | {'固定 PnL':<12} | {'L4 PnL':<12} | {'L4 交易数':<10} | {'L4 平均仓位':<12} | {'差异':<10}")
    print("-" * 80)
    
    total_fixed = 0
    total_l4 = 0
    for res in results:
        diff = res['l4_pnl'] - res['fixed_pnl']
        total_fixed += res['fixed_pnl']
        total_l4 += res['l4_pnl']
        icon = "✅" if diff > 0 else "❌"
        print(f"{res['symbol']:<6} | ${res['fixed_pnl']:<11.2f} | ${res['l4_pnl']:<11.2f} | {res['l4_trades']:<10} | {res['l4_avg_alloc']:<12.1%} | {icon} {diff:+.2f}")
        
    print("-" * 80)
    print(f"总计   | ${total_fixed:<11.2f} | ${total_l4:<11.2f} | {'-':<10} | {'-':<12} | {total_l4 - total_fixed:+.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()
    
    run_l4_backtest(args.days)
