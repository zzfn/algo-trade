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

class TradingBot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        # 使用模拟盘 (paper=True)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.engine = StrategyEngine()
        self.ny_tz = pytz.timezone("America/New_York")
        self.MAX_POSITIONS = 5  # 最大持仓数限制
        self.ALLOCATION_PER_TRADE = 0.10  # 每笔交易分配的资金比例 (10%)
        
    def get_account_info(self):
        return self.trading_client.get_account()

    def get_positions(self):
        return self.trading_client.get_all_positions()

    def get_open_orders(self):
        req = GetOrdersRequest(status=OrderStatus.OPEN)
        return self.trading_client.get_orders(req)

    def run_iteration(self):
        target_dt = datetime.now(self.ny_tz).replace(tzinfo=None)
        print(f"\n" + "="*50)
        print(f"📊 Iteration: {target_dt.strftime('%Y-%m-%d %H:%M:%S')} ET")
        print("="*50)
        
        # 1. 检查账户与持仓
        account = self.get_account_info()
        print(f"Equity: ${float(account.equity):.2f} | Buying Power: ${float(account.buying_power):.2f}")
        
        positions = self.get_positions()
        print(f"📦 Current Positions ({len(positions)}/{self.MAX_POSITIONS}):")
        if not positions:
            print("   (No active positions)")
        for p in positions:
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            print(f"   - {p.symbol}: {p.qty} shares | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        
        # 2. 运行预测模型
        results = self.engine.analyze(target_dt)
        if results.get('l2_ranked') is None or results['l2_ranked'].empty:
            print("❌ No strategy data available.")
            return

        l1_safe = results.get('l1_safe', False)
        l3_signals = results.get('l3_signals', pd.DataFrame())
        all_ranked = results.get('l2_ranked', pd.DataFrame())

        if l3_signals.empty:
            print("❌ No signal data available.")
            return

        # 3. 交易信号执行逻辑
        # 获取多空概率最高的标的
        best_long = l3_signals.sort_values('long_p', ascending=False).iloc[0]
        best_short = l3_signals.sort_values('short_p', ascending=False).iloc[0]

        # 多头信号
        if l1_safe and best_long['long_p'] > 0.45:
            self.execute_trade(best_long['symbol'], OrderSide.BUY, "long", all_ranked, price=best_long['close'])
        elif not l1_safe:
            print("⚠️ L1 Market Safety: UNSAFE (Skipping Longs)")

        # 空头信号
        if best_short['short_p'] > 0.45:
            self.execute_trade(best_short['symbol'], OrderSide.SELL, "short", all_ranked, price=best_short['close'])

    def execute_trade(self, symbol, side, direction, l2_ranked, price):
        # 1. 检查持仓数限制
        positions = self.get_positions()
        if len(positions) >= self.MAX_POSITIONS:
            # 只有当该标的已有持仓时才允许（用于可能的调仓或止损，但目前 logic 是跳过）
            if not any(p.symbol == symbol for p in positions):
                print(f"⚠️ 已达到最大持仓数 ({self.MAX_POSITIONS})，跳过 {symbol}")
                return

        # 2. 检查是否已有该标的持仓
        for p in positions:
            if p.symbol == symbol:
                print(f"ℹ️ {symbol} 已有持仓，忽略信号。")
                return
        
        # 3. 检查是否有该标的的挂单
        open_orders = self.get_open_orders()
        for order in open_orders:
            if order.symbol == symbol:
                print(f"⏳ {symbol} 已有挂单 (ID: {order.id})，等待成交。")
                return

        # 4. 获取 L4 风控参数
        risk = self.engine.get_risk_params(symbol, direction, l2_ranked)
        if not risk:
            return

        tp_pct = risk['tp_pct']
        sl_pct = risk['sl_pct']
        
        # 5. 计算下单股数 (Position Sizing)
        account = self.get_account_info()
        equity = float(account.equity)
        target_value = equity * self.ALLOCATION_PER_TRADE
        qty = int(target_value / price)
        
        if qty <= 0:
            print(f"⚠️ 资金不足以买入 1 股 {symbol} (需要约 ${price:.2f}, 分配额度 ${target_value:.2f})")
            return

        # 计算具体位
        if direction == "long":
            tp_price = round(price * (1 + tp_pct), 2)
            sl_price = round(price * (1 + sl_pct), 2)
        else: # short
            tp_price = round(price * (1 - tp_pct), 2)
            sl_price = round(price * (1 - sl_pct), 2)

        print(f"🚀 触发 {direction.upper()} 信号: {symbol} | 现价: ${price:.2f} | 股数: {qty}")
        print(f"   目标止盈: ${tp_price} ({tp_pct:+.2%})")
        print(f"   目标止损: ${sl_price} ({sl_pct:+.2%})")

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
            print(f"✅ 订单已提交! ID: {order.id}")
        except Exception as e:
            print(f"❌ 下单失败: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=15, help="检查间隔（分钟）")
    args = parser.parse_args()

    bot = TradingBot()
    print(f"✨ 交易机器人启动 | 状态: 实盘自动交易 (模拟盘) | 间隔: {args.interval}min")
    
    while True:
        try:
            bot.run_iteration()
        except Exception as e:
            print(f"Error in iteration: {e}")
        
        print(f"Waiting for {args.interval} minutes...")
        time.sleep(args.interval * 60)

if __name__ == "__main__":
    main()
