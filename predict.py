import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import argparse
from dotenv import load_dotenv
from models.engine import StrategyEngine

def run_hierarchical_prediction():
    load_dotenv()
    
    # ---------------------------------------------------------
    # 参数解析
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="四层架构量化交易预测系统")
    parser.add_argument("--date", type=str, help="指定预测时刻 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM)", default=None)
    args = parser.parse_args()

    engine = StrategyEngine()
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
    
    results = engine.analyze(target_dt)
    
    # L1 Result
    print(f"\n[L1: 市场择时分析] ...")
    status_icon = "🟢" if results.get('l1_safe') else "🔴"
    print(f"{status_icon} 市场环境置信度: {results.get('l1_prob', 0):.1%}")
    if not results.get('l1_safe'):
        print("⚠️ 市场目前处于不安全/弱势区域，L2/L3 仅供参考或建议空头仓位。")
    else:
        print("✅ 市场环境安全，正在进行选股分析...")

    # L2 Result
    print("\n[L2: 标的筛选分析] ...")
    all_ranked = results.get('l2_ranked')
    if all_ranked is None or all_ranked.empty:
        print("❌ 无法获取 L2 选股所需的历史数据。")
        return
    
    print(f"🕒 基于 K 线时刻: {results.get('l2_timestamp')}")
    print("-" * 50)
    print(f"{'排名':<4} | {'代码':<8} | {'价格':<10} | {'相对强度得分'}")
    print("-" * 50)
    for i, (_, row) in enumerate(all_ranked.iterrows()):
        icon = "📈" if row['rank_score'] > 0 else "📉"
        print(f"{i+1:<4} | {row['symbol']:<8} | {row['close']:<10.2f} | {row['rank_score']:.4f} {icon}")

    # L3 Result
    print("\n[L3: 执行信号检测] (针对所有标的)...")
    l3_latest = results.get('l3_signals')
    if l3_latest is None or l3_latest.empty:
        print("❌ 无法获取 L3 信号所需的历史数据。")
        return
        
    print(f"🕒 基于 K 线时刻: {results.get('l3_timestamp')}")
    print("-" * 70)
    print("{:<8} | {:<15} | {:<15} | {:<15}".format("代码", "做多置信度", "做空置信度", "洗盘检测"))
    print("-" * 70)
    for _, row in l3_latest.iterrows():
        shake_desc = "None"
        if row['shakeout_bull'] == 1: shake_desc = "🐮 Bullish Shakeout"
        if row['shakeout_bear'] == 1: shake_desc = "🐻 Bearish Trap"
        print(f"{row['symbol']:<8} | {row['long_p']:<15.2%} | {row['short_p']:<15.2%} | {shake_desc}")

    # L4 Analysis & Summary
    print("\n[L4: 风控建议计算] ...")
    print("\n" + "="*70)
    print("分析总结 (L1 + L2 + L3 + L4)")
    print("="*70)

    best_long = l3_latest.sort_values('long_p', ascending=False).iloc[0]
    best_short = l3_latest.sort_values('short_p', ascending=False).iloc[0]
    found_signal = False
    
    # 做多建议 L4
    if results.get('l1_safe') and best_long['long_p'] > 0.45:
        risk = engine.get_risk_params(best_long['symbol'], "long", all_ranked)
        if risk:
            tp_pct = risk['tp_pct']
            sl_pct = risk['sl_pct']
            curr_price = best_long['close']
            print(f"🚀 [做多建议] 代码: {best_long['symbol']} | L3 置信度: {best_long['long_p']:.1%}")
            print(f"   入场参考价: ${curr_price:.2f}")
            print(f"   止盈目标位: ${curr_price * (1 + tp_pct):.2f} ({tp_pct:+.2%})")
            print(f"   止损触发位: ${curr_price * (1 + sl_pct):.2f} ({sl_pct:+.2%})")
            denom = abs(sl_pct) if abs(sl_pct) > 1e-6 else 1e-6
            print(f"   盈亏比估算: {abs(tp_pct/denom):.2f}:1")
            found_signal = True

    # 做空建议 L4
    if best_short['short_p'] > 0.45:
        if found_signal: print("-" * 40)
        risk = engine.get_risk_params(best_short['symbol'], "short", all_ranked)
        if risk:
            tp_pct = risk['tp_pct']
            sl_pct = risk['sl_pct']
            curr_price = best_short['close']
            print(f"📉 [做空建议] 代码: {best_short['symbol']} | L3 置信度: {best_short['short_p']:.1%}")
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
