
import argparse
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame
from models.engine import StrategyEngine
from models.constants import L2_SYMBOLS, L2_LOOKBACK_DAYS, get_feature_columns
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("l2_backtest")

def run_l2_backtest(days=90, top_n=3):
    logger.info(f"🚀 开始 L2 (选股) 回测, 回溯 {days} 天, Top {top_n}")
    
    engine = StrategyEngine()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    fetch_start = start_date - timedelta(days=L2_LOOKBACK_DAYS)
    
    # 1. 获取数据 (所有 L2 标的)
    logger.info(f"获取 {len(L2_SYMBOLS)} 个标的数据...")
    # 使用 1h 数据进行选股 (模拟每小时或每日重平衡，这里假设每日做一次 rank)
    # 为了简化，我们只在每天收盘时做一次 Rank，持有到第二天收盘
    
    all_dfs = []
    for sym in L2_SYMBOLS:
        df = engine.provider.fetch_bars(sym, TimeFrame.Hour, fetch_start, end_date)
        if not df.empty:
            df = engine.l2_builder.add_all_features(df, is_training=False)
            all_dfs.append(df)
            
    if not all_dfs:
        logger.error("无数据")
        return
        
    full_df = pd.concat(all_dfs)
    
    # 2. 预测 Rank
    logger.info("计算 Rank 分数...")
    # 只取这就按周期内的数据
    test_df = full_df[full_df['timestamp'] >= start_date].copy()
    
    if test_df.empty:
        logger.error("测试区间无数据")
        return

    # 批量预测以加速
    cols = get_feature_columns(test_df)
    test_df['rank_score'] = engine.l2_model.predict(test_df[cols])
    
    # 3. 每日模拟
    # 按天聚合，取每天最后一个小时的数据作为"截面"进行选股
    test_df['date'] = test_df['timestamp'].dt.date
    dates = sorted(test_df['date'].unique())
    
    portfolio_value = 10000.0 # 初始净值
    history = []
    
    logger.info("开始按日回测...")
    
    prev_date = None
    
    for i in range(len(dates) - 1):
        curr_date = dates[i]
        next_date = dates[i+1]
        
        # 获取当日(curr_date)收盘前的截面数据
        day_df = test_df[test_df['date'] == curr_date]
        # 取每个 symbol 当天最后一条记录
        dataset = day_df.sort_values('timestamp').groupby('symbol').tail(1)
        
        # 选股
        ranked = dataset.sort_values('rank_score', ascending=False)
        top_picks = ranked.head(top_n)['symbol'].tolist()
        
        # 计算次日收益
        # 获取选中的股票在 next_date 的收益
        # 简单计算: (Next Close - Curr Close) / Curr Close
        # 更严谨: (Next Open -> Next Close) 或者 (Curr Close -> Next Close)
        # 这里假设: Curr Close 买入, Next Close 卖出 (持有一天)
        
        daily_pnl = 0.0
        
        next_day_df = test_df[test_df['date'] == next_date]
        next_dataset = next_day_df.sort_values('timestamp').groupby('symbol').tail(1)
        
        positions = 0
        for sym in top_picks:
            try:
                curr_price = dataset[dataset['symbol'] == sym]['close'].values[0]
                # 找到次日价格
                next_rows = next_dataset[next_dataset['symbol'] == sym]
                if next_rows.empty:
                    continue
                next_price = next_rows['close'].values[0]
                
                ret = (next_price - curr_price) / curr_price
                daily_pnl += ret
                positions += 1
            except Exception as e:
                pass
        
        avg_ret = daily_pnl / positions if positions > 0 else 0
        portfolio_value *= (1 + avg_ret)
        
        history.append({
            'date': next_date, 
            'value': portfolio_value, 
            'daily_ret': avg_ret,
            'picks': top_picks
        })
    
    # 结果
    res_df = pd.DataFrame(history)
    total_ret = res_df['value'].iloc[-1] / 10000.0 - 1
    
    print("\n" + "="*60)
    print("📊 L2 选股模型 (Ranker) 回测结果")
    print("="*60)
    print(f"回测区间: {dates[0]} ~ {dates[-1]}")
    print(f"选股策略: 每日收盘持有 Top {top_n}")
    print(f"累计收益: {total_ret:+.2%}")
    print(f"日均收益: {res_df['daily_ret'].mean():+.2%}")
    
    print("-" * 60)
    print("最近 5 天持仓与收益:")
    print(res_df.tail())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--top_n", type=int, default=3)
    args = parser.parse_args()
    
    run_l2_backtest(args.days, args.top_n)
