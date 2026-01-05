import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import argparse
from data.provider import DataProvider
from features.macro import L1FeatureBuilder
from features.technical import FeatureBuilder
from models.trainer import SklearnClassifierTrainer, RankingModelTrainer, SignalClassifierTrainer, RiskModelTrainer
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

def run_hierarchical_prediction():
    load_dotenv()
    
    # ---------------------------------------------------------
    # 参数解析
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="四层架构量化交易预测系统")
    parser.add_argument("--date", type=str, help="指定预测时刻 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM)", default=None)
    args = parser.parse_args()

    provider = DataProvider()
    ny_tz = pytz.timezone("America/New_York")
    
    if args.date:
        try:
            if len(args.date) <= 10:
                target_dt = datetime.strptime(args.date, "%Y-%m-%d")
            else:
                target_dt = datetime.strptime(args.date, "%Y-%m-%d %H:%M")
            print(f"💡 使用指定历史时刻进行分析: {target_dt} ET")
        except ValueError:
            print(f"❌ 日期格式错误: {args.date}。请使用 YYYY-MM-DD 或 YYYY-MM-DD HH:MM")
            return
    else:
        target_dt = datetime.now(ny_tz).replace(tzinfo=None)
    
    print("\n" + "="*70)
    print("四层架构交易系统 (L1 -> L2 -> L3 -> L4) | 分析时刻: {dt} ET".format(dt=target_dt.strftime('%Y-%m-%d %H:%M:%S')))
    print("="*70)
    
    # ---------------------------------------------------------
    # L1: Market Timing
    # ---------------------------------------------------------
    print("\n[L1: 市场择时分析] ...")
    l1_builder = L1FeatureBuilder()
    l1_trainer = SklearnClassifierTrainer()
    l1_model = l1_trainer.load("models/artifacts/l1_market_timing.joblib")
    
    # 获取宏观数据
    l1_symbols = ['SPY', 'VIXY', 'TLT']
    l1_start = target_dt - timedelta(days=300) # 需要 MA200
    df_l1_dict = {sym: provider.fetch_bars(sym, TimeFrame.Day, l1_start, target_dt + timedelta(days=1)) for sym in l1_symbols}
    df_l1_feats = l1_builder.build_l1_features(df_l1_dict)
    
    # 选取最接近 target_dt 的一条数据
    df_l1_feats = df_l1_feats[df_l1_feats['timestamp'] <= target_dt]
    if df_l1_feats.empty:
        print("❌ 无法获取 L1 择时所需的历史宏观数据。")
        return
        
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
    l2_model = l2_trainer.load("models/artifacts/l2_stock_selection.joblib")
    
    l2_start = target_dt - timedelta(days=60)
    df_l2_raw = provider.fetch_bars(l2_symbols, TimeFrame.Hour, l2_start, target_dt + timedelta(days=1))
    df_l2_feats = l2_builder.add_all_features(df_l2_raw, is_training=False)
    
    # 筛选有效的截面数据 (target_dt 之前最后一次完整小时线)
    l2_valid = df_l2_feats[df_l2_feats['timestamp'] <= target_dt]
    if l2_valid.empty:
        print("❌ 无法获取 L2 选股所需的历史数据。")
        return
        
    last_h_ts = l2_valid['timestamp'].max()
    l2_latest = l2_valid[l2_valid['timestamp'] == last_h_ts].copy()
    
    l2_exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                  'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                  'max_future_return', 'target_signal', 'local_high', 'local_low']
    l2_features = [c for c in l2_latest.columns if c not in l2_exclude]
    
    l2_latest['rank_score'] = l2_model.predict(l2_latest[l2_features])
    top_stocks = l2_latest.sort_values('rank_score', ascending=False).head(3)
    bottom_stocks = l2_latest.sort_values('rank_score', ascending=True).head(3)
    
    print(f"🕒 基于 K 线时刻: {last_h_ts}")
    print("-" * 50)
    print("📈 做多建议 (Top 3):")
    print(f"{'排名':<4} | {'代码':<8} | {'价格':<10} | {'相对强度得分'}")
    print("-" * 50)
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        print(f"{i+1:<4} | {row['symbol']:<8} | {row['close']:<10.2f} | {row['rank_score']:.4f}")
    
    print("\n" + "-" * 50)
    print("📉 做空建议 (Bottom 3):")
    print(f"{'排名':<4} | {'代码':<8} | {'价格':<10} | {'相对强度得分'}")
    print("-" * 50)
    for i, (_, row) in enumerate(bottom_stocks.iterrows()):
        print(f"{i+1:<4} | {row['symbol']:<8} | {row['close']:<10.2f} | {row['rank_score']:.4f}")
    
    # ---------------------------------------------------------
    # L3: Execution Signal
    # ---------------------------------------------------------
    print("\n[L3: 执行信号检测] (针对 Top 3)...")
    l3_trainer = SignalClassifierTrainer()
    l3_model = l3_trainer.load("models/artifacts/l3_execution.joblib")
    
    top_3_symbols = top_stocks['symbol'].tolist()
    l3_start = target_dt - timedelta(days=10)
    df_l3_raw = provider.fetch_bars(top_3_symbols, TimeFrame(15, TimeFrameUnit.Minute), l3_start, target_dt + timedelta(days=1))
    df_l3_feats = l2_builder.add_all_features(df_l3_raw, is_training=False)
    
    # 确定 target_dt 之前最后完整 15m 线
    l3_valid = df_l3_feats[df_l3_feats['timestamp'] <= target_dt]
    if l3_valid.empty:
        print("❌ 无法获取 L3 信号所需的历史数据。")
        return
        
    last_15m_ts = l3_valid['timestamp'].max()
    l3_latest = l3_valid[l3_valid['timestamp'] == last_15m_ts].copy()
    
    # L3 特征排除 (保留洗盘信号)
    l3_exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                  'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                  'max_future_return', 'target_signal', 'local_high', 'local_low']
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
    
    # ---------------------------------------------------------
    # L4: Risk Management Integration
    # ---------------------------------------------------------
    print("\n[L4: 风控建议计算] ...")
    l4_trainer = RiskModelTrainer()
    l4_tp_long = l4_trainer.load("models/artifacts/l4_risk_tp_long.joblib", "tp_long")
    l4_sl_long = l4_trainer.load("models/artifacts/l4_risk_sl_long.joblib", "sl_long")
    l4_tp_short = l4_trainer.load("models/artifacts/l4_risk_tp_short.joblib", "tp_short")
    l4_sl_short = l4_trainer.load("models/artifacts/l4_risk_sl_short.joblib", "sl_short")

    print("\n" + "="*70)
    print("分析总结 (L1 + L2 + L3 + L4)")
    print("="*70)
    
    if l3_latest.empty:
        print("⚠️ 无法获取 L3 信号数据。")
        return

    best_long = l3_latest.sort_values('long_p', ascending=False).iloc[0]
    best_short = l3_latest.sort_values('short_p', ascending=False).iloc[0]

    found_signal = False
    
    # 做多建议 L4
    if is_safe and best_long['long_p'] > 0.45:
        symbol = best_long['symbol']
        feat_row = l2_latest[l2_latest['symbol'] == symbol]
        if not feat_row.empty:
            tp_pct = l4_tp_long.predict(feat_row[l2_features])[0]
            sl_pct = l4_sl_long.predict(feat_row[l2_features])[0]
            curr_price = best_long['close']
            
            print(f"🚀 [做多建议] 代码: {symbol} | L3 置信度: {best_long['long_p']:.1%}")
            print(f"   入场参考价: ${curr_price:.2f}")
            print(f"   止盈目标位: ${curr_price * (1 + tp_pct):.2f} ({tp_pct:+.2%})")
            print(f"   止损触发位: ${curr_price * (1 + sl_pct):.2f} ({sl_pct:+.2%})")
            denom = abs(sl_pct) if abs(sl_pct) > 1e-6 else 1e-6
            print(f"   盈亏比估算: {abs(tp_pct/denom):.2f}:1")
            found_signal = True

    # 做空建议 L4
    if best_short['short_p'] > 0.45:
        if found_signal: print("-" * 40)
        symbol = best_short['symbol']
        feat_row = l2_latest[l2_latest['symbol'] == symbol]
        if not feat_row.empty:
            tp_pct = l4_tp_short.predict(feat_row[l2_features])[0]
            sl_pct = l4_sl_short.predict(feat_row[l2_features])[0]
            curr_price = best_short['close']
            
            print(f"📉 [做空建议] 代码: {symbol} | L3 置信度: {best_short['short_p']:.1%}")
            print(f"   入场参考价: ${curr_price:.2f}")
            print(f"   止盈目标位: ${curr_price * (1 - tp_pct):.2f} (预期下跌 {tp_pct:.2%})")
            print(f"   止损触发位: ${curr_price * (1 - sl_pct):.2f} (预期上涨 {-sl_pct:.2%})")
            denom = abs(sl_pct) if abs(sl_pct) > 1e-6 else 1e-6
            print(f"   盈亏比估算: {abs(tp_pct/denom):.2f}:1")
            found_signal = True

    if not found_signal:
        print("💡 当前无高置信度入场信号，建议等待或关注洗盘/SMC 结构确认。")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_hierarchical_prediction()
