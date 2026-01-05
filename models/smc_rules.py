"""
SMC (Smart Money Concepts) 规则引擎

基于价格行为和 SMC 概念计算止盈止损点位。
"""

import pandas as pd
from models.constants import RISK_REWARD_RATIO, STOP_LOSS_BUFFER


def get_smc_stop_loss(row: pd.Series, direction: str) -> float:
    """
    基于 SMC 计算止损价格。
    
    做多：止损设在最近 Swing Low 下方
    做空：止损设在最近 Swing High 上方
    
    Args:
        row: 包含 local_high, local_low, close 的 Series
        direction: 'long' 或 'short'
        
    Returns:
        止损价格
    """
    if direction == "long":
        # 做多止损在 Swing Low 下方
        swing_low = row.get('local_low', row['close'] * 0.98)
        return swing_low * (1 - STOP_LOSS_BUFFER)
    else:
        # 做空止损在 Swing High 上方
        swing_high = row.get('local_high', row['close'] * 1.02)
        return swing_high * (1 + STOP_LOSS_BUFFER)


def get_smc_take_profit(row: pd.Series, direction: str, risk_reward: float = None) -> float:
    """
    基于盈亏比计算止盈价格。
    
    止盈 = 入场价 ± (风险距离 × 盈亏比)
    
    Args:
        row: 包含 close 等价格数据的 Series
        direction: 'long' 或 'short'
        risk_reward: 盈亏比，默认使用 RISK_REWARD_RATIO
        
    Returns:
        止盈价格
    """
    if risk_reward is None:
        risk_reward = RISK_REWARD_RATIO
        
    entry = row['close']
    stop_loss = get_smc_stop_loss(row, direction)
    risk = abs(entry - stop_loss)
    
    if direction == "long":
        return entry + risk * risk_reward
    else:
        return entry - risk * risk_reward


def get_smc_risk_params(row: pd.Series, direction: str, risk_reward: float = None) -> dict:
    """
    获取完整的 SMC 风控参数。
    
    Args:
        row: 特征数据行
        direction: 'long' 或 'short'
        risk_reward: 盈亏比
        
    Returns:
        dict: 包含 entry, stop_loss, take_profit, sl_pct, tp_pct 的字典
    """
    entry = row['close']
    stop_loss = get_smc_stop_loss(row, direction)
    take_profit = get_smc_take_profit(row, direction, risk_reward)
    
    if direction == "long":
        sl_pct = (stop_loss / entry) - 1  # 负值
        tp_pct = (take_profit / entry) - 1  # 正值
    else:
        sl_pct = (stop_loss / entry) - 1  # 正值 (做空时止损在上方)
        tp_pct = (take_profit / entry) - 1  # 负值 (做空时止盈在下方)
    
    return {
        'entry': entry,
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'risk_reward': abs(tp_pct / sl_pct) if sl_pct != 0 else 0
    }
