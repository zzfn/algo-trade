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
from models.engine import StrategyEngine
from models.constants import MAX_POSITIONS, TOP_N_TRADES
from utils.logger import setup_logger
from models.constants import SIGNAL_THRESHOLD

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
        # 使用统一配置常量
        self.MAX_POSITIONS = MAX_POSITIONS
        self.TOP_N_TRADES = TOP_N_TRADES
        
    def get_account_info(self):
        return self.trading_client.get_account()

    def get_positions(self):
        return self.trading_client.get_all_positions()

    def get_open_orders(self):
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return self.trading_client.get_orders(req)

    def run_iteration(self):
        target_dt = datetime.now(self.ny_tz).replace(tzinfo=None)
        logger.info("\n" + "="*50)
        logger.info(f"📊 Iteration: {target_dt.strftime('%Y-%m-%d %H:%M:%S')} ET")
        logger.info("="*50)
        
        # 1. 检查账户与持仓
        account = self.get_account_info()
        logger.info(f"Equity: ${float(account.equity):.2f} | Buying Power: ${float(account.buying_power):.2f}")
        
        positions = self.get_positions()
        logger.info(f"📦 Current Positions ({len(positions)}/{self.MAX_POSITIONS}):")
        if not positions:
            logger.info("   (No active positions)")
        for p in positions:
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            logger.info(f"   - {p.symbol}: {p.qty} shares | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        
        # 2. 运行预测模型
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

        # 3. 趋势确认执行逻辑 (Top N 分散交易)
        # 使用 engine.filter_signals 统一过滤高置信度标的
        long_signals = self.engine.filter_signals(l3_signals, direction="long", top_n=self.TOP_N_TRADES)
        short_signals = self.engine.filter_signals(l3_signals, direction="short", top_n=self.TOP_N_TRADES)

        # 4. 持仓管理 (动态止盈止损 / 信号平仓)
        self.manage_positions(l3_signals)

        # 5. 信号执行 (Signal Execution)
        # 根据最新信号与当前持仓状态，决定是保持、开仓还是反手 (反手需在 manage_positions 平仓后下一轮触发)
        
        # 多头信号 (遍历过滤后的标的)
        # [Debug] 跳过 L1 限制，强制执行多头
        if not l1_safe:
            logger.warning("⚠️ L1 Market Safety: UNSAFE (Ignoring check)")

        executed_longs = 0
        for _, signal in long_signals.iterrows():
            success = self.execute_trade(signal['symbol'], OrderSide.BUY, "long", all_ranked, price=signal['close'])
            if success:
                executed_longs += 1
        if executed_longs > 0:
            logger.info(f"📊 本轮多头交易: 成功执行 {executed_longs} 笔")

        # 空头信号 (遍历过滤后的标的)
        executed_shorts = 0
        for _, signal in short_signals.iterrows():
            success = self.execute_trade(signal['symbol'], OrderSide.SELL, "short", all_ranked, price=signal['close'])
            if success:
                executed_shorts += 1
        if executed_shorts > 0:
            logger.info(f"📊 本轮空头交易: 成功执行 {executed_shorts} 笔")

    def manage_positions(self, l3_signals):
        """
        主动管理现有持仓：
        1. 信号反转 -> 立即平仓 (Exit)
        1. 信号反转 -> 立即平仓 (Exit)
        """
        positions = self.get_positions()
        if not positions:
            return

        logger.info(f"🔄 正在检查 {len(positions)} 个持仓的动态管理...")
        


        for p in positions:
            symbol = p.symbol
            qty = abs(int(p.qty))
            side = OrderSide.SELL if p.side == 'long' else OrderSide.BUY # 平仓方向
            entry_price = float(p.avg_entry_price)
            current_price = float(p.current_price)
            
            # --- 1. 信号反转检查 ---
            # 查找该标的的最新 L3 信号
            l3_row = l3_signals[l3_signals['symbol'] == symbol]
            if l3_row.empty:
                continue
            
            l3_data = l3_row.iloc[0]
            should_close = False
            reason = ""

            if p.side == 'long':
                # 持有多头，但出现了强烈的空头信号
                if l3_data['short_p'] > SIGNAL_THRESHOLD:
                    should_close = True
                    reason = f"信号反转 (Short Prob {l3_data['short_p']:.2f} > {SIGNAL_THRESHOLD})"
            else: # short
                # 持有空头，但出现了强烈的多头信号
                if l3_data['long_p'] > SIGNAL_THRESHOLD:
                    should_close = True
                    reason = f"信号反转 (Long Prob {l3_data['long_p']:.2f} > {SIGNAL_THRESHOLD})"
            
            if should_close:
                logger.warning(f"🚨 触发主动平仓: {symbol} | 原因: {reason}")
                try:
                    # 1. 先取消该标的的所有挂单 (释放 held_for_orders)
                    all_orders = self.get_open_orders()
                    for o in all_orders:
                        if o.symbol == symbol:
                            self.trading_client.cancel_order_by_id(o.id)
                            logger.info(f"   - 已撤单: {o.id}")
                    
                    # 2. 执行平仓
                    self.trading_client.close_position(symbol)
                    logger.info(f"✅ 已执行退出 (Exit) {symbol}")
                except Exception as e:
                    logger.error(f"❌ 退出失败 (Exit Failed) {symbol}: {e}")
            


    def execute_trade(self, symbol, side, direction, l2_ranked, price):
        """执行交易，返回 True 表示成功执行，False 表示跳过"""
        # 1. 检查持仓数限制 (Disabled)
        # positions = self.get_positions()
        # if len(positions) >= self.MAX_POSITIONS:
        #     # 只有当该标的已有持仓时才允许（用于可能的调仓或止损，但目前 logic 是跳过）
        #     if not any(p.symbol == symbol for p in positions):
        #         logger.warning(f"⚠️ 已达到最大持仓数 ({self.MAX_POSITIONS})，跳过 {symbol}")
        #         return False

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

        # 5. 计算下单股数 (Position Sizing) - 动态仓位分配
        predicted_return = self.engine.predict_return(symbol, l2_ranked)
        allocation = self.engine.get_allocation(symbol, l2_ranked)
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=1, help="检查间隔（分钟）")
    parser.add_argument("--log-file", type=str, default=None, help="日志文件路径")
    args = parser.parse_args()

    # 如果指定了日志文件，重新配置
    if args.log_file:
        setup_logger("trade", log_file=args.log_file)

    bot = TradingBot()
    logger.info(f"✨ 交易机器人启动 | 状态: 实盘自动交易 (模拟盘) | 间隔: {args.interval}min")
    
    while True:
        try:
            bot.run_iteration()
        except Exception as e:
            logger.error(f"Error in iteration: {e}")
        
        logger.info(f"Waiting for {args.interval} minutes...")
        time.sleep(args.interval * 60)

if __name__ == "__main__":
    main()
