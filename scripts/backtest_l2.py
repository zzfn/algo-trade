
import argparse
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from strategies.engine import StrategyEngine
from config.settings import L2_SYMBOLS, L2_LOOKBACK_DAYS, get_feature_columns
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("l2_backtest")

def run_l2_backtest(days=90, top_n=3):
    logger.info(f"🚀 开始 L2 (选股) 回测, 回溯 {days} 天, Top/Bottom {top_n}")
    
    engine = StrategyEngine()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    fetch_start = start_date - timedelta(days=L2_LOOKBACK_DAYS)
    
    # 1. 获取数据 (所有 L2 标的) - 批量查询
    logger.info(f"获取 {len(L2_SYMBOLS)} 个标的数据...")
    
    # ✅ 批量获取所有 L2 标的数据 (一次性查询)
    df_all = engine.provider.fetch_bars(
        L2_SYMBOLS,  # 批量查询列表
        TimeFrame(5, TimeFrameUnit.Minute), 
        fetch_start, 
        end_date,
        use_redis=True  # 启用 Redis 缓存
    )
    
    # 按标的分组并添加特征
    all_dfs = []
    if not df_all.empty:
        grouped = df_all.groupby('symbol')
        for sym, df in grouped:
            df = engine.l2_builder.add_all_features(df, is_training=False)
            all_dfs.append(df)
            
    if not all_dfs:
        logger.error("无数据")
        return
        
    full_df = pd.concat(all_dfs)
    
    # 2. 预测 Rank
    logger.info("计算 Rank 分数...")
    test_df = full_df[full_df['timestamp'] >= start_date].copy()
    
    if test_df.empty:
        logger.error("测试区间无数据")
        return

    # 批量预测
    cols = get_feature_columns(test_df)
    test_df['rank_score'] = engine.l2_model.predict(test_df[cols])
    
    # 3. 每日模拟 (T日收盘预测 -> T+1日 开盘进 -> T+1日 收盘出)
    test_df['date'] = test_df['timestamp'].dt.date
    dates = sorted(test_df['date'].unique())
    
    # 初始资金分配
    initial_balance = 10000.0
    balance = initial_balance
    
    history = []
    
    logger.info("开始按日回测 (Long Top N vs Short Bottom N)...")
    logger.info("交易模式: T日收盘信号 -> T+1日 Open开仓 -> T+1日 Close平仓 (日内)")
    
    for i in range(len(dates) - 1):
        curr_date = dates[i]   # Signal Date
        next_date = dates[i+1] # Execution Date
        
        # --- Signal Generation (Day T Close) ---
        day_df = test_df[test_df['date'] == curr_date]
        # 取每个 symbol 当天最后一条记录作为"收盘决策点"
        dataset = day_df.sort_values('timestamp').groupby('symbol').tail(1)
        
        # Rank
        ranked = dataset.sort_values('rank_score', ascending=False)
        symbols = ranked['symbol'].tolist()
        
        if len(symbols) < top_n * 2:
            continue
            
        long_picks = symbols[:top_n]
        short_picks = symbols[-top_n:]
        
        # --- Execution (Day T+1 Intraday) ---
        next_day_df = test_df[test_df['date'] == next_date]
        
        daily_long_ret = 0.0
        daily_short_ret = 0.0
        long_count = 0
        short_count = 0
        
        # Calculate Long Returns
        for sym in long_picks:
            sym_df = next_day_df[next_day_df['symbol'] == sym].sort_values('timestamp')
            if sym_df.empty: continue
            
            # Open at first bar, Close at last bar
            open_price = sym_df.iloc[0]['open']
            close_price = sym_df.iloc[-1]['close']
            
            ret = (close_price - open_price) / open_price
            daily_long_ret += ret
            long_count += 1
            
        # Calculate Short Returns (Selling at Open, Buying back at Close)
        for sym in short_picks:
            sym_df = next_day_df[next_day_df['symbol'] == sym].sort_values('timestamp')
            if sym_df.empty: continue
            
            open_price = sym_df.iloc[0]['open']
            close_price = sym_df.iloc[-1]['close']
            
            # Short Return: (Open - Close) / Open
            ret = (open_price - close_price) / open_price
            daily_short_ret += ret
            short_count += 1
            
        avg_long = daily_long_ret / long_count if long_count > 0 else 0
        avg_short = daily_short_ret / short_count if short_count > 0 else 0
        
        # 假设 50/50 资金分配 (不做杠杆，Long和Short各占一半仓位)
        # 或者 Long 100% + Short 100% (Market Neutral 杠杆)? 
        # 简单起见: Total Ret = (Avg Long + Avg Short) / 2
        total_ret = (avg_long + avg_short) / 2
        
        balance *= (1 + total_ret)
        
        history.append({
            'date': next_date, 
            'value': balance, 
            'daily_ret': total_ret,
            'long_ret': avg_long,
            'short_ret': avg_short,
            'longs': long_picks,
            'shorts': short_picks
        })
    
    # 结果
    if not history:
        logger.warning("无交易记录")
        return

    res_df = pd.DataFrame(history)
    total_ret = res_df['value'].iloc[-1] / initial_balance - 1
    
    print("\n" + "="*60)
    print("📊 L2 选股模型 (Ranker) 回测结果")
    print("="*60)
    print(f"回测区间: {dates[0]} ~ {dates[-1]}")
    print(f"模式: T+1 Open -> Close (日内), Long Top {top_n} & Short Bottom {top_n}")
    print(f"累计收益: {total_ret:+.2%}")
    print(f"日均收益: {res_df['daily_ret'].mean():+.2%}")
    print(f"日均做多: {res_df['long_ret'].mean():+.2%}")
    print(f"日均做空: {res_df['short_ret'].mean():+.2%}")
    
    print("-" * 60)
    print("最近 5 天绩效:")
    print(res_df[['date', 'value', 'daily_ret', 'long_ret', 'short_ret']].tail())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--top_n", type=int, default=3)
    args = parser.parse_args()
    
    run_l2_backtest(args.days, args.top_n)
