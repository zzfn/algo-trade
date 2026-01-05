import os
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderClass
from models.engine import StrategyEngine
from models.constants import MAX_POSITIONS, TOP_N_TRADES
from utils.logger import setup_logger

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
        req = GetOrdersRequest(status=OrderStatus.OPEN)
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

        # 多头信号 (遍历过滤后的标的)
        if l1_safe:
            executed_longs = 0
            for _, signal in long_signals.iterrows():
                success = self.execute_trade(signal['symbol'], OrderSide.BUY, "long", all_ranked, price=signal['close'])
                if success:
                    executed_longs += 1
            if executed_longs > 0:
                logger.info(f"📊 本轮多头交易: 成功执行 {executed_longs} 笔")
        else:
            logger.warning("⚠️ L1 Market Safety: UNSAFE (Skipping Longs)")

        # 空头信号 (遍历过滤后的标的)
        executed_shorts = 0
        for _, signal in short_signals.iterrows():
            success = self.execute_trade(signal['symbol'], OrderSide.SELL, "short", all_ranked, price=signal['close'])
            if success:
                executed_shorts += 1
        if executed_shorts > 0:
            logger.info(f"📊 本轮空头交易: 成功执行 {executed_shorts} 笔")

    def execute_trade(self, symbol, side, direction, l2_ranked, price):
        """执行交易，返回 True 表示成功执行，False 表示跳过"""
        # 1. 检查持仓数限制
        positions = self.get_positions()
        if len(positions) >= self.MAX_POSITIONS:
            # 只有当该标的已有持仓时才允许（用于可能的调仓或止损，但目前 logic 是跳过）
            if not any(p.symbol == symbol for p in positions):
                logger.warning(f"⚠️ 已达到最大持仓数 ({self.MAX_POSITIONS})，跳过 {symbol}")
                return False

        # 2. 检查是否已有该标的持仓
        for p in positions:
            if p.symbol == symbol:
                logger.info(f"ℹ️ {symbol} 已有持仓，忽略信号。")
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
        tp_price = risk['take_profit']
        sl_price = risk['stop_loss']
        
        logger.info(f"🎯 {symbol} | 入场: ${price:.2f} | 止盈: ${tp_price:.2f} ({risk['tp_pct']:.2%}) | 止损: ${sl_price:.2f} ({risk['sl_pct']:.2%})")
            sl_price = round(price * (1 + sl_pct), 2)
        else: # short
            tp_price = round(price * (1 - tp_pct), 2)
            sl_price = round(price * (1 - sl_pct), 2)

        logger.info(f"🚀 触发 {direction.upper()} 信号: {symbol} | 现价: ${price:.2f} | 股数: {qty}")
        logger.info(f"   目标止盈: ${tp_price} ({tp_pct:+.2%})")
        logger.info(f"   目标止损: ${sl_price} ({sl_pct:+.2%})")

        try:
            # 构造 Bracket Order (支架订单: 包含自动止盈止损)
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty, 
                side=side,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price)
            )
            order = self.trading_client.submit_order(order_data)
            logger.info(f"✅ 订单已提交! ID: {order.id}")
            return True
        except Exception as e:
            logger.error(f"❌ 下单失败: {e}")
            return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=15, help="检查间隔（分钟）")
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
