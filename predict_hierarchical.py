import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data.provider import DataProvider
from features.l1_builder import L1FeatureBuilder
from features.builder import FeatureBuilder
from models.trainer import SklearnClassifierTrainer, RankingModelTrainer, SignalClassifierTrainer
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

def run_hierarchical_prediction():
    load_dotenv()
    provider = DataProvider()
    ny_tz = pytz.timezone("America/New_York")
    target_dt = datetime.now(ny_tz).replace(tzinfo=None)
    
    print("\n" + "="*70)
    print("三层架构交易系统 (L1 -> L2 -> L3) | 分析时刻: {dt} ET".format(dt=target_dt.strftime('%Y-%m-%d %H:%M:%S')))
    print("="*70)
    
    # ---------------------------------------------------------
    # L1: Market Timing
    # ---------------------------------------------------------
    print("\n[L1: 市场择时分析] ...")
    l1_builder = L1FeatureBuilder()
    l1_trainer = SklearnClassifierTrainer()
    l1_model = l1_trainer.load("models/l1_market_timing.joblib")
    
    # 获取宏观数据
    l1_symbols = ['SPY', 'VIXY', 'TLT']
    l1_start = target_dt - timedelta(days=300) # 需要 MA200
    df_l1_dict = {sym: provider.fetch_bars(sym, TimeFrame.Day, l1_start, target_dt + timedelta(days=1)) for sym in l1_symbols}
    df_l1_feats = l1_builder.build_l1_features(df_l1_dict)
    
    latest_l1 = df_l1_feats.iloc[-1:]
    l1_features = ['spy_return_1d', 'spy_dist_ma200', 'vixy_level', 'vixy_change_1d', 'tlt_return_5d']
    
    market_safe_prob = l1_model.predict_proba(latest_l1[l1_features])[0][1]
    is_safe = market_safe_prob > 0.5
    
    status_icon = "🟢" if is_safe else "🔴"
    print(f"{status_icon} 市场环境置信度: {market_safe_prob:.1%}")
    if not is_safe:
        print("⚠️ 市场目前处于不安全/弱势区域，L2/L3 仅供参考或建议空头仓位。")
    else:
        print("✅ 市场环境安全，正在进行选股分析...")

    # ---------------------------------------------------------
    # L2: Stock Selection
    # ---------------------------------------------------------
    print("\n[L2: 标的筛选分析] ...")
    l2_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC']
    l2_builder = FeatureBuilder()
    l2_trainer = RankingModelTrainer()
    l2_model = l2_trainer.load("models/l2_stock_selection.joblib")
    
    l2_start = target_dt - timedelta(days=60)
    df_l2_raw = provider.fetch_bars(l2_symbols, TimeFrame.Hour, l2_start, target_dt + timedelta(days=1))
    df_l2_feats = l2_builder.add_all_features(df_l2_raw, is_training=False)
    
    # 筛选有效的截面数据 (最后一次完整小时线)
    df_l2_feats['is_complete'] = (df_l2_feats['timestamp'] + timedelta(hours=1)) <= target_dt
    last_h_ts = df_l2_feats[df_l2_feats['is_complete']]['timestamp'].max()
    l2_latest = df_l2_feats[df_l2_feats['timestamp'] == last_h_ts].copy()
    
    # L2 特征排除 (不包括洗盘信号，因为 L2 训练时还没加)
    l2_exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                  'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                  'max_future_return', 'target_signal', 'local_high', 'local_low', 'is_complete',
                  'shakeout_bull', 'shakeout_bear']
    l2_features = [c for c in l2_latest.columns if c not in l2_exclude]
    
    l2_latest['rank_score'] = l2_model.predict(l2_latest[l2_features])
    top_stocks = l2_latest.sort_values('rank_score', ascending=False).head(3)
    
    print(f"🕒 基于 K 线时刻: {last_h_ts}")
    print("-" * 50)
    print(f"{'排名':<4} | {'代码':<8} | {'价格':<10} | {'相对强度得分'}")
    print("-" * 50)
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        print(f"{i+1:<4} | {row['symbol']:<8} | {row['close']:<10.2f} | {row['rank_score']:.4f}")
    
    # ---------------------------------------------------------
    # L3: Execution Signal
    # ---------------------------------------------------------
    print("\n[L3: 执行信号检测] (针对 Top 3)...")
    l3_trainer = SignalClassifierTrainer()
    l3_model = l3_trainer.load("models/l3_execution.joblib")
    
    top_3_symbols = top_stocks['symbol'].tolist()
    l3_start = target_dt - timedelta(days=10)
    df_l3_raw = provider.fetch_bars(top_3_symbols, TimeFrame(15, TimeFrameUnit.Minute), l3_start, target_dt + timedelta(days=1))
    df_l3_feats = l2_builder.add_all_features(df_l3_raw, is_training=False)
    
    # 确定最后完整 15m 线
    df_l3_feats['is_complete'] = (df_l3_feats['timestamp'] + timedelta(minutes=15)) <= target_dt
    last_15m_ts = df_l3_feats[df_l3_feats['is_complete']]['timestamp'].max()
    l3_latest = df_l3_feats[df_l3_feats['timestamp'] == last_15m_ts].copy()
    
    # L3 特征排除 (保留洗盘信号)
    l3_exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                  'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                  'max_future_return', 'target_signal', 'local_high', 'local_low', 'is_complete']
    l3_features = [c for c in l3_latest.columns if c not in l3_exclude]
    
    print(f"输入 L3 特征维度: {len(l3_features)}")
    probs = l3_model.predict_proba(l3_latest[l3_features])
    l3_latest['long_p'] = probs[:, 1]
    l3_latest['short_p'] = probs[:, 2]
    
    print(f"🕒 基于 K 线时刻: {last_15m_ts}")
    print("-" * 70)
    print("{:<8} | {:<15} | {:<15} | {:<15}".format("代码", "做多置信度", "做空置信度", "洗盘检测"))
    print("-" * 70)
    for _, row in l3_latest.iterrows():
        shake_desc = "None"
        if row['shakeout_bull'] == 1: shake_desc = "🐮 Bullish Shakeout"
        if row['shakeout_bear'] == 1: shake_desc = "🐻 Bearish Trap"
        
        print(f"{row['symbol']:<8} | {row['long_p']:<15.2%} | {row['short_p']:<15.2%} | {shake_desc}")
    
    print("\n" + "="*70)
    print("分析总结: ", end="")
    best_candidate = l3_latest.sort_values('long_p', ascending=False).iloc[0]
    if is_safe and best_candidate['long_p'] > 0.45:
        print(f"🚀 核心推荐 [{best_candidate['symbol']}] 做多。")
    else:
        print("💡 当前无高置信度入场信号，建议等待或关注洗盘反抽。")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_hierarchical_prediction()
