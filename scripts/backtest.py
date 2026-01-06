import os
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from data.provider import DataProvider
from features.macro import L1FeatureBuilder
from features.technical import FeatureBuilder
from models.engine import StrategyEngine
from models.constants import (
    get_feature_columns, 
    L1_SAFE_THRESHOLD, SIGNAL_THRESHOLD, 
    TOP_N_TRADES
)
from utils.logger import setup_logger

logger = setup_logger("backtest")
load_dotenv()

class Position:
    def __init__(self, symbol, direction, entry_price, size, tp_price, sl_price, entry_time):
        self.symbol = symbol
        self.direction = direction  # 'long' or 'short'
        self.entry_price = float(entry_price)
        self.size = int(size)
        self.tp_price = float(tp_price)
        self.sl_price = float(sl_price)
        self.entry_time = entry_time
        self.status = 'open'  # open, closed
        self.exit_price = 0.0
        self.exit_time = None
        self.exit_reason = None # tp, sl, signal_reversal, time_exit
        self.pnl = 0.0
        self.return_pct = 0.0

    def update(self, current_bar, params=None):
        """
        检查是否触发离场条件。
        模拟盘中 High/Low 触发。
        """
        if self.status != 'open':
            return

        high = current_bar['high']
        low = current_bar['low']
        close = current_bar['close']
        ts = current_bar['timestamp']

        # 1. 检查止损 (优先检查)
        stop_triggered = False
        if self.direction == 'long':
            if low <= self.sl_price:
                stop_triggered = True
                exec_price = self.sl_price  # 假设刚好在止损价成交 (略乐观，忽略滑点)
                # 如果开盘直接低开在止损价下方，则以开盘价止损
                if current_bar['open'] < self.sl_price:
                    exec_price = current_bar['open']
        else: # short
            if high >= self.sl_price:
                stop_triggered = True
                exec_price = self.sl_price
                if current_bar['open'] > self.sl_price:
                    exec_price = current_bar['open']
        
        if stop_triggered:
            self.close(exec_price, ts, 'stop_loss')
            return

        # 2. 检查止盈
        take_profit_triggered = False
        if self.direction == 'long':
            if high >= self.tp_price:
                take_profit_triggered = True
                exec_price = self.tp_price
                if current_bar['open'] > self.tp_price:
                    exec_price = current_bar['open']
        else: # short
            if low <= self.tp_price:
                take_profit_triggered = True
                exec_price = self.tp_price
                if current_bar['open'] < self.tp_price:
                    exec_price = current_bar['open']

        if take_profit_triggered:
            self.close(exec_price, ts, 'take_profit')
            return
            
    def close(self, price, time, reason):
        self.status = 'closed'
        self.exit_price = float(price)
        self.exit_time = time
        self.exit_reason = reason
        
        if self.direction == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.size
            self.return_pct = (self.exit_price / self.entry_price) - 1
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.size
            self.return_pct = 1 - (self.exit_price / self.entry_price)

class BacktestEngine:
    def __init__(self, initial_equity=100000.0, top_n=TOP_N_TRADES):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.cash = initial_equity
        self.positions = []  # 活跃持仓
        self.closed_positions = [] # 历史持仓
        self.history = [] # 每日净值记录
        
        self.engine = StrategyEngine() # 复用 StrategyEngine 加载模型
        self.top_n = top_n
        
        # 缓存数据
        self.bars = {} # symbol -> dataframe
        self.market_features = None # L1 dataframe
        
    def run(self, symbols, timeframe, start_date, end_date):
        logger.info(f"🚀 开始回测: {start_date} ~ {end_date} | 初始资金: ${self.initial_equity:,.2f}")
        
        # 1. 预加载和预处理数据
        self._prepare_data(symbols, timeframe, start_date, end_date)
        
        # 2. 生成时间轴 (按分钟/小时对齐)
        timeline = sorted(list(set(t for df in self.bars.values() for t in df['timestamp'])))
        timeline = [t for t in timeline if t >= start_date]
        
        logger.info(f"⏳ 时间步总数: {len(timeline)}")
        
        # 3. 主循环
        for current_ts in timeline:
            self._process_bar(current_ts)
            
        # 4. 生成报告
        self._generate_report(timeframe)

    def _prepare_data(self, symbols, timeframe, start_date, end_date):
        logger.info("📥 正在预加载数据与特征...")
        
        # L1 数据 (已移除 L1 择时，此处不再加载)
        # l1_start = start_date - timedelta(days=365)
        # df_l1_dict = {sym: self.engine.provider.fetch_bars(sym, TimeFrame.Day, l1_start, end_date) for sym in self.engine.l1_symbols}
        # self.market_features = self.engine.l1_builder.build_l1_features(df_l1_dict)
        
        # L2/3/4 数据 (这里简化：直接用主回测周期的数据计算特征，假设 timeframe 足够细)
        # 注意：这里我们直接用传入的 timeframe 作为主驱动周期
        fetch_start = start_date - timedelta(days=60) # 预留指标计算窗口
        
        for sym in symbols:
            df = self.engine.provider.fetch_bars([sym], timeframe, fetch_start, end_date)
            if not df.empty:
                # 预计算所有特征 (L2/L3/L4 需要的)
                df = self.engine.l2_builder.add_all_features(df, is_training=False)
                # 预计算 L2 得分 (提速)
                cols = get_feature_columns(df)
                df['rank_score'] = self.engine.l2_model.predict(df[cols])
                
                # 预计算 L3 概率 (已移除 L3 信号过滤，跳过计算)
                # probs = self.engine.l3_model.predict_proba(df[cols])
                # df['long_p'] = probs[:, 1]
                # df['short_p'] = probs[:, 2]
                
                # 为了后续逻辑兼容，填充 dummy 值
                df['long_p'] = 0.99 
                df['short_p'] = 0.99
                
                # 仅保留回测所需列以节省内存，但在回测时需要用整行数据计算 risk
                # df = df[['timestamp', 'open', 'high', 'low', 'close', 'rank_score', 'long_p', 'short_p', ...]]
                self.bars[sym] = df
        
        logger.info(f"✅ 数据准备完成。覆盖 {len(self.bars)} 个标的。")

    def _process_bar(self, current_ts):
        # 1. 获取当前时刻的所有标的数据
        current_bars = {}
        for sym, df in self.bars.items():
            # 找到当前时刻或最近的前一个时刻的数据 (Forward Fill)
            # 这里简单处理：只取精确匹配当前时刻的数据
            row = df[df['timestamp'] == current_ts]
            if not row.empty:
                current_bars[sym] = row.iloc[0]
        
        if not current_bars:
            return

        # 2. 检查现有持仓 (止盈止损)
        active_positions = []
        for pos in self.positions:
            if pos.symbol in current_bars:
                # 更新状态 (检查 SL/TP)
                pos.update(current_bars[pos.symbol])
                
                if pos.status == 'closed':
                    self.cash += pos.exit_price * pos.size
                    self.closed_positions.append(pos)
                    logger.debug(f"平仓 {pos.symbol} ({pos.direction}): {pos.exit_reason} | PnL: ${pos.pnl:.2f} ({pos.return_pct:.2%})")
                else:
                    active_positions.append(pos)
            else:
                active_positions.append(pos) # 数据缺失，保持持仓不变
        self.positions = active_positions

        # 3. 市场环境判断 (L1) - SIMPLIFIED: 始终假设安全
        is_safe = True

        # 4. 信号生成与开仓 (仅当现金充足)
        if self.cash > 0:
            # 收集所有标的的 L2 rank 和 L3 signal
            candidates = []
            for sym, bar in current_bars.items():
                # 过滤掉已有持仓的标的
                if any(p.symbol == sym for p in self.positions):
                    continue
                
                # 简化版逻辑：只看 rank_score
                # Top N 个做多，Bottom N 个做空 (如果 rank_score 足够低)

                candidates.append({'sym': sym, 'dir': 'long', 'score': bar['rank_score'], 'bar': bar})
                # 同时加入做空候选 (score 取反，用于统一排序 - score越低做空优先级越高)
                candidates.append({'sym': sym, 'dir': 'short', 'score': -bar['rank_score'], 'bar': bar})
            
            # 按置信度排序，取 Top N
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_picks = candidates[:self.top_n]
            
            for pick in top_picks:
                if self.cash <= 0:
                    break
                    
                sym = pick['sym']
                direction = pick['dir']
                bar = pick['bar']
                price = bar['close']
                
                # 5. 风控参数与动态仓位 (L4 + SMC)
                # 构造一个临时的 DataFrame 用于 L4 预测 (需要特征列)
                # bar 是 Series, 转为 DataFrame
                l2_df = pd.DataFrame([bar])
                
                # 动态仓位
                allocation = self.engine.get_allocation(sym, l2_df)
                # SMC 止盈止损
                risk = self.engine.get_risk_params(sym, direction, l2_df)
                
                if risk:
                    target_value = self.equity * allocation
                    size = int(target_value / price)
                    
                    if size > 0 and (size * price) <= self.cash:
                        new_pos = Position(
                            sym, direction, price, size, 
                            risk['take_profit'], risk['stop_loss'], current_ts
                        )
                        self.positions.append(new_pos)
                        self.cash -= price * size
                        logger.debug(f"开仓 {sym} ({direction}): ${price:.2f} | 仓位 {allocation:.1%} | TP: {risk['take_profit']} | SL: {risk['stop_loss']}")

        # 6. 更新净值记录
        current_equity = self.cash
        for pos in self.positions:
            # 使用当前收盘价估算浮动净值
            if pos.symbol in current_bars:
                curr_price = current_bars[pos.symbol]['close']
                if pos.direction == 'long':
                    val = curr_price * pos.size
                else:
                    # 做空净值计算: 初始市值 + 浮动盈亏
                    # 简化：做空时借入股票卖出，现金增加，负债增加。
                    # 这里用：开仓时现金已扣除(作为保证金)，此处加回 (Entry + PnL)
                    val = (pos.entry_price * pos.size) + (pos.entry_price - curr_price) * pos.size
                current_equity += val
            else:
                # 缺失数据时沿用入场成本估值（保守）
                current_equity += pos.entry_price * pos.size
                
        self.history.append({'timestamp': current_ts, 'equity': current_equity, 'cash': self.cash})
        self.equity = current_equity

    def _generate_report(self, timeframe):
        print("\n" + "="*80)
        print("🏁 回测完成. 生成报告...")
        print("="*80)
        
        df_hist = pd.DataFrame(self.history).set_index('timestamp')
        if df_hist.empty:
            print("❌ 无回测数据")
            return
            
        # 计算基础指标
        total_ret = (self.equity / self.initial_equity) - 1
        days = (df_hist.index[-1] - df_hist.index[0]).days
        annual_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
        
        # 最大回撤
        roll_max = df_hist['equity'].cummax()
        dd = df_hist['equity'] / roll_max - 1
        mdd = dd.min()
        
        # 显示当前持仓
        current_holdings_pnl = 0.0
        if self.positions:
            print(f"\n💼 当前持仓 ({len(self.positions)}):")
            print(f"{'代码':<6} | {'方向':<6} | {'入场价':<10} | {'当前价':<10} | {'浮动盈亏':<10} | {'回报率':<8}")
            print("-" * 70)
            for p in self.positions:
                # 获取该标的最后已知价格
                last_price = p.entry_price # 默认 (如果没有更新数据)
                if p.symbol in self.bars:
                     # 尝试拿最后一根 K 线的收盘价
                    last_price = self.bars[p.symbol].iloc[-1]['close']
                
                if p.direction == 'long':
                    unrealized_pnl = (last_price - p.entry_price) * p.size
                    ret = (last_price / p.entry_price) - 1
                else:
                    unrealized_pnl = (p.entry_price - last_price) * p.size
                    ret = 1 - (last_price / p.entry_price)
                
                current_holdings_pnl += unrealized_pnl
                icon = "🟢" if unrealized_pnl > 0 else "🔴"
                print(f"{p.symbol:<6} | {p.direction:<6} | ${p.entry_price:<9.2f} | ${last_price:<9.2f} | {icon} ${unrealized_pnl:<8.2f} | {ret:<+7.2%}")
            
            print(f"当前持仓总浮亏: ${current_holdings_pnl:.2f}")

        # 交易统计
        all_closed = self.closed_positions
        if not all_closed:
            print("⚠️ 期间无平仓交易")
        else:
            wins = [p for p in all_closed if p.pnl > 0]
            losses = [p for p in all_closed if p.pnl <= 0]
            win_rate = len(wins) / len(all_closed)
            avg_win = np.mean([p.pnl for p in wins]) if wins else 0
            avg_loss = np.mean([p.pnl for p in losses]) if losses else 0
            profit_factor = abs(sum(p.pnl for p in wins) / sum(p.pnl for p in losses)) if losses else float('inf')
            
            print(f"📊 资金表现:")
            print(f"  初始资金: ${self.initial_equity:,.2f}")
            print(f"  最终权益: ${self.equity:,.2f} ({total_ret:+.2%})")
            print(f"  年化收益: {annual_ret:+.2%}")
            print(f"  最大回撤: {mdd:.2%}")
            
            print(f"\n🎯 交易统计:")
            print(f"  总交易数: {len(all_closed)}")
            print(f"  胜率:     {win_rate:.1%} ({len(wins)} 胜 / {len(losses)} 负)")
            print(f"  盈亏比:   {profit_factor:.2f}")
            print(f"  平均盈利: ${avg_win:.2f}")
            print(f"  平均亏损: ${avg_loss:.2f}")
            
            # 显示最近 5 笔交易
            print(f"\n📝 最近 10 笔交易记录:")
            print(f"{'时间':<20} | {'代码':<5} | {'方向':<5} | {'PnL':<10} | {'回报率':<8} | {'原因'}")
            print("-" * 80)
            for p in all_closed[-10:]:
                icon = "🟢" if p.pnl > 0 else "🔴"
                print(f"{str(p.exit_time):<20} | {p.symbol:<5} | {p.direction:<5} | {icon} ${p.pnl:<8.2f} | {p.return_pct:<+7.2%} | {p.exit_reason}")
                
        print("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timeframe", nargs="?", default="1h")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", help="如 AAPL,TSLA")
    parser.add_argument("--top_n", type=int, default=TOP_N_TRADES)
    args = parser.parse_args()
    
    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        # 默认使用所有 L2 标的
        from models.constants import L2_SYMBOLS
        symbols = L2_SYMBOLS
        
    start_date = datetime.now() - timedelta(days=args.days)
    end_date = datetime.now()
    
    # 转换 timeframe
    tf_map = {'1h': TimeFrame.Hour, '15m': TimeFrame(15, TimeFrameUnit.Minute), '1d': TimeFrame.Day}
    tf = tf_map.get(args.timeframe, TimeFrame.Hour)
    
    engine = BacktestEngine(top_n=args.top_n)
    engine.run(symbols, tf, start_date, end_date)

if __name__ == "__main__":
    main()
