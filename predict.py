import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import argparse
from dotenv import load_dotenv
from models.engine import StrategyEngine
from models.constants import TOP_N_TRADES, SIGNAL_THRESHOLD
from utils.logger import setup_logger

# 初始化日志
logger = setup_logger("predict")

def run_hierarchical_prediction():
    load_dotenv()
    
    # ---------------------------------------------------------
    # 参数解析
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="四层架构量化交易预测系统")
    parser.add_argument("--date", type=str, help="指定预测时刻 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM)", default=None)
    parser.add_argument("--log-file", type=str, default=None, help="日志文件路径")
    args = parser.parse_args()

    # 如果指定了日志文件，重新配置
    if args.log_file:
        setup_logger("predict", log_file=args.log_file)

    engine = StrategyEngine()
    ny_tz = pytz.timezone("America/New_York")
    
    if args.date:
        try:
            if len(args.date) <= 10:
                target_dt = datetime.strptime(args.date, "%Y-%m-%d")
            else:
                target_dt = datetime.strptime(args.date, "%Y-%m-%d %H:%M")
            logger.info(f"💡 使用指定历史时刻进行分析: {target_dt} ET")
        except ValueError:
            logger.error(f"❌ 日期格式错误: {args.date}。请使用 YYYY-MM-DD 或 YYYY-MM-DD HH:MM")
            return
    else:
        target_dt = datetime.now(ny_tz).replace(tzinfo=None)
    
    logger.info("\n" + "="*70)
    logger.info("四层架构交易系统 (L1 -> L2 -> L3 -> L4) | 分析时刻: {dt} ET".format(dt=target_dt.strftime('%Y-%m-%d %H:%M:%S')))
    logger.info("="*70)
    
    results = engine.analyze(target_dt)
    
    # L1 Result
    logger.info(f"\n[L1: 市场择时分析] ...")
    status_icon = "🟢" if results.get('l1_safe') else "🔴"
    logger.info(f"{status_icon} 市场环境置信度: {results.get('l1_prob', 0):.1%}")
    if not results.get('l1_safe'):
        logger.warning("⚠️ 市场目前处于不安全/弱势区域，L2/L3 仅供参考或建议空头仓位。")
    else:
        logger.info("✅ 市场环境安全，正在进行选股分析...")

    # L2 Result
    logger.info("\n[L2: 标的筛选分析] ...")
    all_ranked = results.get('l2_ranked')
    if all_ranked is None or all_ranked.empty:
        logger.error("❌ 无法获取 L2 选股所需的历史数据。")
        return
    
    logger.info(f"🕒 基于 K 线时刻: {results.get('l2_timestamp')}")
    logger.info("-" * 50)
    logger.info(f"{'排名':<4} | {'代码':<8} | {'价格':<10} | {'相对强度得分'}")
    logger.info("-" * 50)
    for i, (_, row) in enumerate(all_ranked.iterrows()):
        icon = "📈" if row['rank_score'] > 0 else "📉"
        logger.info(f"{i+1:<4} | {row['symbol']:<8} | {row['close']:<10.2f} | {row['rank_score']:.4f} {icon}")

    # L3 Result
    logger.info("\n[L3: 执行信号检测] (针对所有标的)...")
    l3_latest = results.get('l3_signals')
    if l3_latest is None or l3_latest.empty:
        logger.error("❌ 无法获取 L3 信号所需的历史数据。")
        return
        
    logger.info(f"🕒 基于 K 线时刻: {results.get('l3_timestamp')}")
    logger.info("-" * 70)
    logger.info("{:<8} | {:<15} | {:<15} | {:<15}".format("代码", "做多置信度", "做空置信度", "洗盘检测"))
    logger.info("-" * 70)
    for _, row in l3_latest.iterrows():
        shake_desc = "None"
        if row['shakeout_bull'] == 1: shake_desc = "🐮 Bullish Shakeout"
        if row['shakeout_bear'] == 1: shake_desc = "🐻 Bearish Trap"
        logger.info(f"{row['symbol']:<8} | {row['long_p']:<15.2%} | {row['short_p']:<15.2%} | {shake_desc}")

    # L4 Analysis & Summary
    logger.info("\n[L4: 风控建议计算] ...")
    logger.info("\n" + "="*70)
    logger.info(f"分析总结 (L1 + L2 + L3 + L4) - Top {TOP_N_TRADES} 分散交易建议 (置信度阈值: {SIGNAL_THRESHOLD:.0%})")
    logger.info("="*70)
    
    # 使用 engine.filter_signals 统一过滤高置信度标的
    long_signals = engine.filter_signals(l3_latest, direction="long")
    short_signals = engine.filter_signals(l3_latest, direction="short")
    
    # 做多建议
    if results.get('l1_safe') and not long_signals.empty:
        logger.info(f"\n🚀 [做多建议] Top {len(long_signals)} 高置信度标的:")
        logger.info("-" * 60)
        for i, (_, signal) in enumerate(long_signals.iterrows(), 1):
            risk = engine.get_risk_params(signal['symbol'], "long", all_ranked)
            if risk:
                tp_pct = risk['tp_pct']
                sl_pct = risk['sl_pct']
                curr_price = signal['close']
                denom = abs(sl_pct) if abs(sl_pct) > 1e-6 else 1e-6
                rr_ratio = abs(tp_pct/denom)
                logger.info(f"   [{i}] {signal['symbol']}: 置信度 {signal['long_p']:.1%} | 入场 ${curr_price:.2f}")
                logger.info(f"       止盈 ${curr_price * (1 + tp_pct):.2f} ({tp_pct:+.2%}) | 止损 ${curr_price * (1 + sl_pct):.2f} ({sl_pct:+.2%}) | 盈亏比 {rr_ratio:.2f}:1")
    elif not results.get('l1_safe'):
        logger.warning("\n⚠️ 市场环境不安全，跳过做多建议")
    else:
        logger.info("\n💡 无高置信度做多信号")

    # 做空建议
    if not short_signals.empty:
        logger.info(f"\n📉 [做空建议] Top {len(short_signals)} 高置信度标的:")
        logger.info("-" * 60)
        for i, (_, signal) in enumerate(short_signals.iterrows(), 1):
            risk = engine.get_risk_params(signal['symbol'], "short", all_ranked)
            if risk:
                tp_pct = risk['tp_pct']
                sl_pct = risk['sl_pct']
                curr_price = signal['close']
                denom = abs(sl_pct) if abs(sl_pct) > 1e-6 else 1e-6
                rr_ratio = abs(tp_pct/denom)
                logger.info(f"   [{i}] {signal['symbol']}: 置信度 {signal['short_p']:.1%} | 入场 ${curr_price:.2f}")
                logger.info(f"       止盈 ${curr_price * (1 - tp_pct):.2f} (预期下跌 {tp_pct:.2%}) | 止损 ${curr_price * (1 - sl_pct):.2f} (预期上涨 {-sl_pct:.2%}) | 盈亏比 {rr_ratio:.2f}:1")
    else:
        logger.info("\n💡 无高置信度做空信号")

    if long_signals.empty and short_signals.empty:
        logger.info("\n💡 当前无高置信度入场信号，建议等待或关注洗盘/SMC 结构确认。")
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    run_hierarchical_prediction()
