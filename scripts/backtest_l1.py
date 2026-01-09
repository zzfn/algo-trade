
import argparse
import pandas as pd
import numpy as np


from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame
from models.engine import StrategyEngine
from models.constants import L1_SYMBOLS, L1_LOOKBACK_DAYS, L1_SAFE_THRESHOLD
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("l1_backtest")

def run_l1_backtest(days=365):
    logger.info(f"🚀 开始 L1 (市场择时) 回测, 回溯 {days} 天")
    
    engine = StrategyEngine()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    fetch_start = start_date - timedelta(days=L1_LOOKBACK_DAYS) # Extra buffer for MA200
    
    # 1. 获取数据 - 批量查询所有市场指标
    logger.info("获取市场数据 (SPY, VIXY, TLT)...")
    
    # ✅ 批量获取所有市场指标数据 (一次性查询)
    df_all = engine.provider.fetch_bars(
        L1_SYMBOLS,  # 批量查询列表
        TimeFrame.Day, 
        fetch_start, 
        end_date,
        use_redis=True  # 启用 Redis 缓存
    )
    
    # 按标的分组
    df_l1_dict = {}
    if not df_all.empty:
        grouped = df_all.groupby('symbol')
        for sym, df in grouped:
            df_l1_dict[sym] = df
    else:
        logger.error("无法获取市场数据")
        return
    
    # 2. 构建特征
    logger.info("构建 L1 特征...")
    df_features = engine.l1_builder.build_l1_features(df_l1_dict)
    
    # 过滤回测周期
    df_test = df_features[df_features['timestamp'] >= start_date].copy()
    if df_test.empty:
        logger.error("无回测数据 (可能数据不足)")
        return

    # 3. 预测
    logger.info("运行模型预测...")
    feature_cols = ['spy_return_1d', 'spy_dist_ma200', 'vixy_level', 'vixy_change_1d', 'tlt_return_5d']
    probs = engine.l1_model.predict_proba(df_test[feature_cols])[:, 1]
    df_test['prob_safe'] = probs
    df_test['is_safe'] = df_test['prob_safe'] > L1_SAFE_THRESHOLD
    
    # 4. 模拟交易 (持有 SPY vs 空仓)
    # is_safe = True -> 持有 SPY
    # is_safe = False -> 持有现金 (收益为 0，忽略利息)
    
    df_test['spy_ret'] = df_test['close'].pct_change().shift(-1) # 下一天的收益 (T+1)
    # 如果今天判断 is_safe，明天持有
    df_test['strategy_ret'] = np.where(df_test['is_safe'], df_test['spy_ret'], 0.0)
    
    # 5. 计算累计收益
    df_test['cum_spy'] = (1 + df_test['spy_ret']).cumprod()
    df_test['cum_strategy'] = (1 + df_test['strategy_ret']).cumprod()
    
    # 6. 统计指标
    total_spy = df_test['cum_spy'].iloc[-2] - 1 if len(df_test) > 1 else 0
    total_strat = df_test['cum_strategy'].iloc[-2] - 1 if len(df_test) > 1 else 0
    
    print("\n" + "="*60)
    print("📊 L1 市场择时 模型回测结果")
    print("="*60)
    print(f"回测区间: {df_test['timestamp'].min().date()} ~ {df_test['timestamp'].max().date()}")
    print(f"交易天数: {len(df_test)}")
    print(f"SPY 基准收益: {total_spy:+.2%}")
    print(f"L1 策略收益: {total_strat:+.2%}")
    print(f"安全天数占比: {df_test['is_safe'].mean():.1%}")
    
    # Win Rate (正确预测上涨的日子)
    # Define 'Success': Predict Safe AND SPY > 0, or Predict Unsafe AND SPY < 0
    # Note: simplistic view.
    
    print("-" * 60)
    print("最近 5 天预测:")
    print(df_test[['timestamp', 'close', 'prob_safe', 'is_safe', 'spy_ret']].tail())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    
    run_l1_backtest(args.days)
