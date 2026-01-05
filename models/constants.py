"""
模型相关常量定义

统一管理特征排除列表，避免多处重复定义导致的维护问题。
"""

# =============================================================================
# 策略配置常量
# =============================================================================

# --- 信号阈值 ---
L1_SAFE_THRESHOLD = 0.5      # L1 市场安全阈值 (概率 > 此值为安全)
SIGNAL_THRESHOLD = 0.45      # L3 趋势置信度阈值 (做多/做空概率 > 此值触发)

# --- 时间窗口 (天数) ---
L1_LOOKBACK_DAYS = 300       # L1 市场择时回溯天数
L2_LOOKBACK_DAYS = 60        # L2 标的筛选回溯天数
L3_LOOKBACK_DAYS = 10        # L3 趋势确认 (Trend) 回溯天数

# --- 交易参数 ---
TOP_N_TRADES = 3             # 每轮选择 top_n 个高置信度标的进行交易
MAX_POSITIONS = 5            # 最大持仓数限制

# --- 动态仓位分配 (基于预期收益) ---
ALLOCATION_TIERS = [
    (0.05, 0.15),   # 预期收益 > 5% → 分配 15%
    (0.02, 0.10),   # 预期收益 > 2% → 分配 10%
    (0.01, 0.05),   # 预期收益 > 1% → 分配 5%
    (0.00, 0.02),   # 预期收益 > 0% → 分配 2%
]
MIN_ALLOCATION = 0.02        # 最小仓位
MAX_ALLOCATION = 0.15        # 最大仓位

# --- SMC 风控参数 ---
RISK_REWARD_RATIO = 2.0      # 默认盈亏比
STOP_LOSS_BUFFER = 0.005     # 止损位距离 Swing High/Low 的缓冲 (0.5%)

# --- 标的池 ---
L1_SYMBOLS = ['SPY', 'VIXY', 'TLT']  # L1 市场择时标的
L2_SYMBOLS = [                        # L2 交易标的池
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
    'TSLA', 'NVDA', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC'
]

# 特征排除列表 - 这些列不应该作为模型输入特征
FEATURE_EXCLUDE_COLS = [
    # 元数据
    'timestamp', 'symbol',
    # 原始价格数据 (会泄露信息)
    'open', 'high', 'low', 'close', 'volume',
    # 中间计算变量
    'atr', 'vwap', 'trade_count', 'local_high', 'local_low',
    # 目标标签 (需要预测的输出)
    'target_return', 'target_rank', 'target_signal', 'max_future_return',
    'target_future_return',  # L4 收益预测目标
    # L4 旧版风控目标 (已废弃)
    'target_tp_long_pct', 'target_sl_long_pct',
    'target_tp_short_pct', 'target_sl_short_pct',
    # 预测输出
    'rank_score', 'long_p', 'short_p',
]


def get_feature_columns(df):
    """
    从 DataFrame 中提取可用于模型的特征列。
    
    Args:
        df: 包含所有列的 DataFrame
        
    Returns:
        list: 特征列名列表
    """
    return [c for c in df.columns if c not in FEATURE_EXCLUDE_COLS]


def get_allocation_by_return(predicted_return: float) -> float:
    """
    根据预期收益计算仓位分配比例。
    
    Args:
        predicted_return: 预期收益率 (如 0.02 表示 2%)
        
    Returns:
        float: 仓位分配比例
    """
    for threshold, allocation in ALLOCATION_TIERS:
        if predicted_return > threshold:
            return allocation
    return MIN_ALLOCATION

