import os
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderClass, QueryOrderStatus
import numpy as np
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from models.engine import StrategyEngine
from models.constants import TOP_N_TRADES, SIGNAL_THRESHOLD, L1_RISK_FACTOR
from models.dynamic_params import get_dynamic_params
from utils.logger import setup_logger
import json
from pathlib import Path
import redis

# 初始化日志
logger = setup_logger("trade")

class TradingBot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        # 使用模拟盘 (paper=True)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.engine = StrategyEngine()
        self.ny_tz = pytz.timezone("America/New_York")
        #使用统一配置常量

        self.TOP_N_TRADES = TOP_N_TRADES
        
        # 初始化 Redis (用于 Dashboard 状态共享)
        self.redis_client = None
        self.state_file = Path("data/trading_state.json")
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            logger.info(f"✅ Redis 已连接 ({redis_host}:{redis_port})")
        except Exception as e:
            logger.warning(f"⚠️ Redis 连接失败: {e}, Dashboard 将使用文件备份")
        
    def get_account_info(self):
        return self.trading_client.get_account()

    def get_positions(self):
        return self.trading_client.get_all_positions()

    def get_open_orders(self):
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return self.trading_client.get_orders(req)

    def run_iteration(self):
        """
        执行一轮交易检查
        
        Returns:
            datetime or None: 如果市场关闭,返回下次开盘时间;否则返回 None
        """
        target_dt = datetime.now(self.ny_tz).replace(tzinfo=None)
        logger.info("\n" + "="*50)
        logger.info(f"📊 Iteration: {target_dt.strftime('%Y-%m-%d %H:%M:%S')} ET")
        logger.info("="*50)
        
        # 1. 检查市场是否开放
        try:
            clock = self.trading_client.get_clock()
            # 使用 Alpaca 服务器时间作为基准 (避免本地时间错误)
            server_now = clock.timestamp.astimezone(self.ny_tz)
            logger.info(f"⏰ Server Time: {server_now.strftime('%Y-%m-%d %H:%M:%S')} ET")
            
            # 使用服务器时间更新 target_dt
            target_dt = server_now.replace(tzinfo=None)
            
            if not clock.is_open:
                next_open = clock.next_open.astimezone(self.ny_tz)
                next_close = clock.next_close.astimezone(self.ny_tz) if clock.next_close else None
                logger.info(f"⏸️  市场当前关闭")
                logger.info(f"   下次开盘: {next_open.strftime('%Y-%m-%d %H:%M:%S')} ET")
                if next_close:
                    logger.info(f"   下次收盘: {next_close.strftime('%Y-%m-%d %H:%M:%S')} ET")
                logger.info("   跳过本轮交易检查")
                
                # 即使市场关闭，也发布一次状态（显示账户信息）
                try:
                    account = self.get_account_info()
                    positions = self.get_positions()
                    self.publish_state(
                        account=account,
                        positions=positions,
                        long_signals=pd.DataFrame(),  # 空信号
                        short_signals=pd.DataFrame(),
                        l1_safe=False,
                        l1_prob=0.0,
                        is_market_open=False
                    )
                except Exception as e:
                    logger.warning(f"⚠️ 市场关闭时状态发布失败: {e}")
                
                return next_open.replace(tzinfo=None)  # 返回下次开盘时间
            else:
                logger.info(f"✅ 市场开放中 (收盘时间: {clock.next_close.astimezone(self.ny_tz).strftime('%H:%M:%S')} ET)")
        except Exception as e:
            logger.warning(f"⚠️  无法获取市场状态: {e}")
            logger.warning("   继续执行(使用本地时间)...")
            # fallback to local time if clock fails
            target_dt = datetime.now(self.ny_tz).replace(tzinfo=None)
        
        # 2. 检查账户与持仓
        account = self.get_account_info()
        logger.info(f"Equity: ${float(account.equity):.2f} | Buying Power: ${float(account.buying_power):.2f}")
        
        positions = self.get_positions()
        logger.info(f"📦 Current Positions ({len(positions)}):")
        if not positions:
            logger.info("   (No active positions)")
        for p in positions:
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            logger.info(f"   - {p.symbol}: {p.qty} shares | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        
        # 3. 获取实时市场特征并预测 L5 参数
        try:
            # 获取最近 30 天的 SPY 数据
            spy_start = target_dt - timedelta(days=45)
            spy_data = self.engine.provider.fetch_bars(['SPY'], TimeFrame.Day, spy_start, target_dt, use_redis=True)
            
            if not spy_data.empty:
                # 计算市场特征
                spy_returns = spy_data['close'].pct_change()
                spy_return_1d = spy_returns.iloc[-1]
                spy_volatility = spy_returns.std() * np.sqrt(252)
                recent_vol = spy_returns.tail(20).std()
                
                # 简单趋势判断
                sma_20 = spy_data['close'].rolling(20).mean().iloc[-1]
                sma_50 = spy_data['close'].rolling(50).mean().iloc[-1]
                trend = 1 if sma_20 > sma_50 else -1
                
                market_features = {
                    'spy_return_1d': spy_return_1d,
                    'spy_volatility': spy_volatility,
                    'vixy_level': 16.0,  # 默认值 (如果无法获取实时 VIXY)
                    'market_trend': trend,
                    'recent_volatility': recent_vol
                }
                
                # 预测动态参数
                dynamic_params = get_dynamic_params(market_features)
                
                # 更新参数
                current_threshold = dynamic_params['signal_threshold']
                current_top_n = dynamic_params['top_n_trades']
                current_risk_factor = dynamic_params['l1_risk_factor']
                
                logger.info(f"🧠 L5 动态参数预测:")
                logger.info(f"   Threshold: {current_threshold:.3f} (Base: {SIGNAL_THRESHOLD})")
                logger.info(f"   Top N:     {current_top_n} (Base: {TOP_N_TRADES})")
                logger.info(f"   Risk Fx:   {current_risk_factor:.3f} (Base: {L1_RISK_FACTOR})")
            else:
                logger.warning("⚠️ 无法获取 SPY 数据,使用默认参数")
                current_threshold = SIGNAL_THRESHOLD
                current_top_n = TOP_N_TRADES
                current_risk_factor = L1_RISK_FACTOR
                
        except Exception as e:
            logger.warning(f"⚠️ L5 预测失败: {e},使用默认参数")
            current_threshold = SIGNAL_THRESHOLD
            current_top_n = TOP_N_TRADES
            current_risk_factor = L1_RISK_FACTOR

        # 4. 运行预测模型
        results = self.engine.analyze(target_dt)
        if results.get('l2_ranked') is None or results['l2_ranked'].empty:
            logger.error("❌ No strategy data available.")
            return

        l1_safe = results.get('l1_safe', False)
        l3_signals = results.get('l3_signals', pd.DataFrame())
        all_ranked = results.get('l2_ranked', pd.DataFrame())

        if l3_signals.empty:
            logger.error("❌ No signal data available.")
            return
            
        l3_ts = results.get('l3_timestamp')
        if l3_ts:
            logger.info(f"📡 API Data Time: {l3_ts.strftime('%Y-%m-%d %H:%M:%S')} ET")

        # 5. 趋势确认执行逻辑 (Top N 分散交易)
        # 使用 engine.filter_signals 统一过滤高置信度标的 - 传入动态参数
        long_signals = self.engine.filter_signals(l3_signals, direction="long", top_n=current_top_n, threshold=current_threshold)
        short_signals = self.engine.filter_signals(l3_signals, direction="short", top_n=current_top_n, threshold=current_threshold)

        # 5. 持仓管理 (动态止盈止损 / 信号平仓)
        closed_symbols = self.manage_positions(l3_signals, all_ranked)

        # 6. 信号执行 (Signal Execution)
        # L1 作为风险因子: 不安全时降低仓位而非禁止交易
        l1_prob = results.get('l1_prob', 0.0)
        if l1_safe:
            logger.info(f"✅ L1 Market Safety: SAFE (概率: {l1_prob:.2%}) - 使用正常仓位")
        else:
            logger.warning(f"⚠️ L1 Market Safety: UNSAFE (概率: {l1_prob:.2%}) - 降低仓位至 {current_risk_factor:.1%}")

        # 多头信号
        executed_longs = 0
        for _, signal in long_signals.iterrows():
            success = self.execute_trade(signal['symbol'], OrderSide.BUY, "long", all_ranked, price=signal['close'], l1_safe=l1_safe, closed_symbols=closed_symbols)
            if success:
                executed_longs += 1
        if executed_longs > 0:
            logger.info(f"📊 本轮多头交易: 成功执行 {executed_longs} 笔")

        # 空头信号
        executed_shorts = 0
        for _, signal in short_signals.iterrows():
            success = self.execute_trade(signal['symbol'], OrderSide.SELL, "short", all_ranked, price=signal['close'], l1_safe=l1_safe, closed_symbols=closed_symbols)
            if success:
                executed_shorts += 1
        if executed_shorts > 0:
            logger.info(f"📊 本轮空头交易: 成功执行 {executed_shorts} 笔")
        
        # 7. 发布状态到 Dashboard
        self.publish_state(
            account=account,
            positions=positions,
            long_signals=long_signals,
            short_signals=short_signals,
            l1_safe=l1_safe,
            l1_prob=l1_prob,
            is_market_open=True
        )

    def manage_positions(self, l3_signals, l2_ranked):
        """
        主动管理现有持仓:
        1. 基于价格的止盈止损检查 (优先)
        2. 信号反转检查
        
        Args:
            l3_signals: L3 趋势信号 DataFrame
            l2_ranked: L2 排序后的 DataFrame (用于获取特征和计算风控参数)
        """

        positions = self.get_positions()
        closed_symbols = set()
        if not positions:
            return closed_symbols

        logger.info(f"🔄 正在检查 {len(positions)} 个持仓的动态管理...")

        for p in positions:
            symbol = p.symbol
            qty = abs(int(p.qty))
            side = OrderSide.SELL if p.side == 'long' else OrderSide.BUY  # 平仓方向
            entry_price = float(p.avg_entry_price)
            current_price = float(p.current_price)
            
            should_close = False
            reason = ""
            
            # --- 1. 基于价格的止盈止损检查 (优先) ---
            # 从 l2_ranked 获取该标的的特征数据
            feat_row = l2_ranked[l2_ranked['symbol'] == symbol]
            
            if not feat_row.empty:
                # 计算该持仓的止盈止损价格
                direction = 'long' if p.side == 'long' else 'short'
                risk_params = self.engine.get_risk_params(symbol, direction, l2_ranked)
                
                if risk_params:
                    tp_price = risk_params['take_profit']
                    sl_price = risk_params['stop_loss']
                    
                    if p.side == 'long':
                        # 做多: 价格跌破止损或突破止盈
                        if current_price <= sl_price:
                            should_close = True
                            pnl_pct = (current_price / entry_price - 1) * 100
                            reason = f"触发止损 (当前价 ${current_price:.2f} <= 止损价 ${sl_price:.2f}, {pnl_pct:+.2f}%)"
                        elif current_price >= tp_price:
                            should_close = True
                            pnl_pct = (current_price / entry_price - 1) * 100
                            reason = f"触发止盈 (当前价 ${current_price:.2f} >= 止盈价 ${tp_price:.2f}, {pnl_pct:+.2f}%)"
                    else:  # short
                        # 做空: 价格突破止损或跌破止盈
                        if current_price >= sl_price:
                            should_close = True
                            pnl_pct = (1 - current_price / entry_price) * 100
                            reason = f"触发止损 (当前价 ${current_price:.2f} >= 止损价 ${sl_price:.2f}, {pnl_pct:+.2f}%)"
                        elif current_price <= tp_price:
                            should_close = True
                            pnl_pct = (1 - current_price / entry_price) * 100
                            reason = f"触发止盈 (当前价 ${current_price:.2f} <= 止盈价 ${tp_price:.2f}, {pnl_pct:+.2f}%)"
            
            # --- 2. 信号反转检查 (只有在未触发止盈止损时才检查) ---
            if not should_close:
                l3_row = l3_signals[l3_signals['symbol'] == symbol]
                if not l3_row.empty:
                    l3_data = l3_row.iloc[0]
                    
                    if p.side == 'long':
                        # 持有多头,但出现了强烈的空头信号
                        if l3_data['short_p'] > SIGNAL_THRESHOLD:
                            should_close = True
                            reason = f"信号反转 (Short Prob {l3_data['short_p']:.2%} > {SIGNAL_THRESHOLD:.2%})"
                    else:  # short
                        # 持有空头,但出现了强烈的多头信号
                        if l3_data['long_p'] > SIGNAL_THRESHOLD:
                            should_close = True
                            reason = f"信号反转 (Long Prob {l3_data['long_p']:.2%} > {SIGNAL_THRESHOLD:.2%})"
            
            # --- 3. 执行平仓 ---
            if should_close:
                logger.warning(f"🚨 触发主动平仓: {symbol} | 原因: {reason}")
                try:
                    # 1. 先取消该标的的所有挂单
                    all_orders = self.get_open_orders()
                    for o in all_orders:
                        if o.symbol == symbol:
                            self.trading_client.cancel_order_by_id(o.id)
                            logger.info(f"   - 已撤单: {o.id}")
                    
                    # 2. 执行平仓
                    self.trading_client.close_position(symbol)
                    logger.info(f"✅ 已执行平仓: {symbol}")
                    closed_symbols.add(symbol)
                except Exception as e:
                    logger.error(f"❌ 平仓失败 {symbol}: {e}")
        
        return closed_symbols

    def execute_trade(self, symbol, side, direction, l2_ranked, price, l1_safe=True, closed_symbols=None):

        if closed_symbols and symbol in closed_symbols:
            logger.info(f"ℹ️ {symbol} 本轮刚触发平仓，跳过开仓 (等待订单结算)。")
            return False

        positions = self.get_positions()


        # 2. 检查是否已有该标的持仓 (若有，则说明方向一致，继续持有)
        for p in positions:
            if p.symbol == symbol:
                logger.info(f"ℹ️ {symbol} 已有持仓，保持现状 (Hold)。")
                return False
        
        # 3. 检查是否有该标的的挂单
        open_orders = self.get_open_orders()
        for order in open_orders:
            if order.symbol == symbol:
                logger.info(f"⏳ {symbol} 已有挂单 (ID: {order.id})，等待成交。")
                return False

        # 5. 计算下单股数 (Position Sizing) - 动态仓位分配 (考虑 L1 风险)
        predicted_return = self.engine.predict_return(symbol, l2_ranked)
        allocation = self.engine.get_allocation(symbol, l2_ranked, l1_safe=l1_safe)
        
        account = self.get_account_info()
        equity = float(account.equity)
        target_value = equity * allocation
        qty = int(target_value / price)
        
        logger.info(f"💰 {symbol} 预期收益: {predicted_return:.2%}, 分配比例: {allocation:.1%}, 目标股数: {qty}")
        
        if qty <= 0:
            logger.warning(f"⚠️ 资金不足以买入 1 股 {symbol} (需要约 ${price:.2f}, 分配额度 ${target_value:.2f})")
            return False

        # 6. 设置止盈止损价格 (从 SMC 规则获取)
        risk = self.engine.get_risk_params(symbol, direction, l2_ranked)
        if not risk:
            logger.warning(f"⚠️ 无法计算 {symbol} 的风控参数 (可能数据不足)，跳过")
            return False

        tp_price = risk['take_profit']
        sl_price = risk['stop_loss']
        tp_pct = risk['tp_pct']
        sl_pct = risk['sl_pct']

        logger.info(f"🚀 触发 {direction.upper()} 信号: {symbol} | 现价: ${price:.2f} | 股数: {qty}")
        logger.info(f"   [Ref Only] 建议止盈: ${tp_price:.2f} ({tp_pct:+.2%})")
        logger.info(f"   [Ref Only] 建议止损: ${sl_price:.2f} ({sl_pct:+.2%})")

        try:
            # 构造 Market Order (仅市价单，不带止盈止损，依靠实时轮询平仓)
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty, 
                side=side,
                time_in_force=TimeInForce.GTC
            )
            order = self.trading_client.submit_order(order_data)
            logger.info(f"✅ 订单已提交! ID: {order.id}")
            return True
        except Exception as e:
            logger.error(f"❌ 下单失败: {e}")
            return False
    
    def publish_state(self, account, positions, long_signals, short_signals, l1_safe, l1_prob, is_market_open):
        """发布交易状态到 Dashboard (Redis + 文件备份)"""
        try:
            # 构建状态数据
            state = {
                "account": {
                    "equity": float(account.equity),
                    "cash": float(account.cash),
                    "buying_power": float(account.buying_power),
                    "portfolio_value": float(account.portfolio_value) if hasattr(account, 'portfolio_value') else float(account.equity),
                },
                "positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "qty": p.qty,
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "unrealized_pl": float(p.unrealized_pl),
                        "unrealized_plpc": float(p.unrealized_plpc),
                    }
                    for p in positions
                ],
                "signals": {
                    "long": long_signals.to_dict('records') if not long_signals.empty else [],
                    "short": short_signals.to_dict('records') if not short_signals.empty else [],
                },
                "orders": [],  # 可以扩展获取最近订单
                "status": {
                    "last_update": datetime.now(self.ny_tz).isoformat(),
                    "is_market_open": is_market_open,
                    "l1_safe": l1_safe,
                    "l1_prob": l1_prob,
                    "is_running": True,
                }
            }
            
            # 辅助函数: 递归清理 NaN/Inf
            def clean_nan(obj):
                if isinstance(obj, float):
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return obj
                elif isinstance(obj, dict):
                    return {k: clean_nan(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan(v) for v in obj]
                return obj

            # 清理非法浮点数
            state = clean_nan(state)

            # 序列化为 JSON
            state_json = json.dumps(state, default=str)
            
            # 写入 Redis
            if self.redis_client:
                try:
                    self.redis_client.set("trading:state", state_json, ex=300)  # 5分钟过期
                    logger.debug("✅ 状态已发布到 Redis")
                except Exception as e:
                    logger.warning(f"⚠️ Redis 写入失败: {e}")
            
            # 备份到文件
            self.state_file.parent.mkdir(exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
            logger.debug("✅ 状态已备份到文件")
            
        except Exception as e:
            logger.error(f"❌ 状态发布失败: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=1, help="检查间隔(分钟),默认1分钟")
    parser.add_argument("--log-file", type=str, default=None, help="日志文件路径")
    args = parser.parse_args()

    # 如果指定了日志文件，重新配置
    if args.log_file:
        setup_logger("trade", log_file=args.log_file)

    bot = TradingBot()
    logger.info(f"✨ 交易机器人启动 | 状态: 实盘自动交易 (模拟盘) | 间隔: {args.interval}min")
    
    iteration_count = 0
    
    while True:
        try:
            iteration_count += 1
            
            # 执行一轮检查
            next_open = bot.run_iteration()
            
            # 如果市场关闭,智能等待到开盘前5分钟
            if next_open:
                now = datetime.now(bot.ny_tz).replace(tzinfo=None)
                wait_until = next_open - timedelta(minutes=5)  # 提前5分钟唤醒
                wait_seconds = (wait_until - now).total_seconds()
                
                if wait_seconds > 60:  # 如果等待时间超过1分钟
                    logger.info(f"💤 市场关闭,将在 {wait_until.strftime('%Y-%m-%d %H:%M:%S')} ET 唤醒 (开盘前5分钟)")
                    logger.info(f"   等待时长: {wait_seconds/3600:.1f} 小时")
                    time.sleep(max(wait_seconds, 0))
                    continue
            
        except KeyboardInterrupt:
            logger.info("\n⏹️  用户中断,正在安全退出...")
            logger.info("📊 最终持仓状态:")
            try:
                positions = bot.get_positions()
                if positions:
                    for p in positions:
                        pnl = float(p.unrealized_pl)
                        logger.info(f"   - {p.symbol}: {p.qty} shares | PnL: ${pnl:+.2f}")
                else:
                    logger.info("   (无持仓)")
            except:
                pass
            break
            
        except ConnectionError as e:
            logger.error(f"🌐 网络连接错误: {e}")
            logger.info("⏳ 等待 60 秒后重试...")
            time.sleep(60)
            continue
            
        except Exception as e:
            logger.error(f"❌ 运行错误: {e}", exc_info=True)
            logger.info("⏳ 等待下一轮检查...")
        
        # 正常等待间隔
        logger.info(f"\n💤 等待 {args.interval} 分钟...\n")
        time.sleep(args.interval * 60)

if __name__ == "__main__":
    main()
