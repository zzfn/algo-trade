"""
模型相关常量定义

统一管理特征排除列表，避免多处重复定义导致的维护问题。
"""

# =============================================================================
# 策略配置常量
# =============================================================================

# --- 信号阈值 ---
L1_SAFE_THRESHOLD = 0.5      # L1 市场安全阈值 (概率 > 此值为安全)
SIGNAL_THRESHOLD = 0.45      # L3 信号置信度阈值 (做多/做空概率 > 此值触发)

# --- 时间窗口 (天数) ---
L1_LOOKBACK_DAYS = 300       # L1 市场择时回溯天数
L2_LOOKBACK_DAYS = 60        # L2 标的筛选回溯天数
L3_LOOKBACK_DAYS = 10        # L3 执行信号回溯天数

# --- 交易参数 ---
TOP_N_TRADES = 3             # 每轮选择 top_n 个高置信度标的进行交易
MAX_POSITIONS = 5            # 最大持仓数限制
ALLOCATION_PER_TRADE = 0.10  # 每笔交易分配的资金比例 (10%)

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
    # L4 风控目标
    'target_tp_long_pct', 'target_sl_long_pct',
    'target_tp_short_pct', 'target_sl_short_pct',
    # 预测输出
    'rank_score',
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
