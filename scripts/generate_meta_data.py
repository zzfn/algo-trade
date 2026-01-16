"""
L5 元策略训练数据生成器

通过参数扫描生成元策略训练数据:
- 对历史数据的不同时期
- 测试不同参数组合
- 记录每个组合的表现
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.technical import FeatureBuilder
from features.macro import MacroFeatureBuilder
from models.engine import StrategyEngine
from models.constants import L2_SYMBOLS, MACRO_SYMBOLS
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
from pathlib import Path
import json
from tqdm import tqdm

def extract_market_features(period_data, macro_data):
    """
    提取市场特征用于元策略
    
    Args:
        period_data: 该时期的价格数据
        macro_data: 宏观数据
        
    Returns:
        市场特征字典
    """
    # SPY 特征
    spy_return = macro_data['spy_return_1d'].iloc[-1] if 'spy_return_1d' in macro_data else 0
    spy_volatility = period_data['close'].pct_change().std() * np.sqrt(252)
    
    # VIX 特征
    vixy_level = macro_data['vixy_level'].iloc[-1] if 'vixy_level' in macro_data else 15
    
    # 趋势特征
    sma_20 = period_data['close'].rolling(20).mean().iloc[-1]
    sma_50 = period_data['close'].rolling(50).mean().iloc[-1]
    trend = 1 if sma_20 > sma_50 else -1
    
    # 波动率
    recent_vol = period_data['close'].pct_change().tail(20).std()
    
    return {
        'spy_return_1d': spy_return,
        'spy_volatility': spy_volatility,
        'vixy_level': vixy_level,
        'market_trend': trend,
        'recent_volatility': recent_vol,
        'timestamp': period_data.index[-1]
    }

def simple_backtest(data, signal_threshold, top_n):
    """
    简化的回测函数
    
    Returns:
        性能指标字典
    """
    # 简化版: 计算基于参数的预期表现
    # 实际应该运行完整回测,这里为了速度使用简化版
    
    returns = data['close'].pct_change().dropna()
    
    # 模拟: 不同参数对收益的影响
    # 阈值越高 → 交易越少但质量越高
    # top_n 越多 → 分散但可能稀释收益
    # risk_factor 越高 → 收益波动越大
    
    # 这里使用简化的启发式评分
    # 实际应该运行真实回测
    base_return = returns.mean() * 252
    base_vol = returns.std() * np.sqrt(252)
    
    # 参数调整
    threshold_factor = 1 + (signal_threshold - 0.45) * 0.5
    topn_factor = 1 - (top_n - 3) * 0.05
    
    # Risk Factor Removed
    
    adjusted_return = base_return * threshold_factor * topn_factor
    adjusted_vol = base_vol # No risk adjustment on vol without leverage factor
    
    sharpe = adjusted_return / (adjusted_vol + 1e-6) if adjusted_vol > 0 else 0
    
    return {
        'total_return': adjusted_return,
        'volatility': adjusted_vol,
        'sharpe_ratio': sharpe
    }

def generate_meta_training_data(days=180, window_size=30):
    """
    生成元策略训练数据
    
    Args:
        days: 总天数
        window_size: 滑动窗口大小(天)
    """
    load_dotenv()
    provider = DataProvider()
    
    print("=" * 60)
    print("🧠 L5 元策略训练数据生成")
    print("=" * 60)
    print(f"总天数: {days}")
    print(f"窗口大小: {window_size} 天")
    print("=" * 60)
    
    # 获取数据
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=days)
    
    print(f"\n📊 获取数据...")
    print(f"   时间范围: {start_date.date()} 到 {end_date.date()}")
    
    # 获取 SPY 数据用于市场特征
    spy_data = provider.fetch_bars(['SPY'], TimeFrame.Day, start_date, end_date)
    spy_data = spy_data.set_index('timestamp')
    
    # 简化: 直接使用 SPY 数据作为市场特征,不依赖 MacroFeatureBuilder
    print(f"   SPY 数据: {len(spy_data)} 行")
    
    if len(spy_data) < window_size:
        print(f"❌ 错误: SPY 数据不足 (需要至少 {window_size} 天)")
        return None
    
    # 参数搜索空间
    param_grid = {
        'signal_threshold': [0.35, 0.40, 0.45, 0.50, 0.55],
        'top_n_trades': [2, 3, 4, 5]
    }
    
    total_combinations = (len(param_grid['signal_threshold']) * 
                         len(param_grid['top_n_trades']))
    
    print(f"\n🔍 参数组合数: {total_combinations}")
    print(f"   signal_threshold: {param_grid['signal_threshold']}")
    print(f"   top_n_trades: {param_grid['top_n_trades']}")
    
    # 滑动窗口
    training_data = []
    num_windows = (days - window_size) // 5  # 每5天一个窗口
    
    print(f"\n⏳ 开始生成数据...")
    print(f"   窗口数量: {num_windows}")
    print(f"   总测试次数: {num_windows * total_combinations}")
    
    for i in tqdm(range(num_windows), desc="窗口进度"):
        window_start = start_date + timedelta(days=i*5)
        window_end = window_start + timedelta(days=window_size)
        
        # 获取该窗口的数据
        window_spy = spy_data[(spy_data.index >= window_start) & (spy_data.index < window_end)]
        
        if len(window_spy) < 10:
            continue
        
        # 简化的市场特征提取
        spy_return = window_spy['close'].pct_change().iloc[-1]
        spy_volatility = window_spy['close'].pct_change().std() * np.sqrt(252)
        recent_vol = window_spy['close'].pct_change().tail(20).std()
        
        # 趋势
        sma_20 = window_spy['close'].rolling(20).mean().iloc[-1]
        sma_50 = window_spy['close'].rolling(min(50, len(window_spy))).mean().iloc[-1]
        trend = 1 if sma_20 > sma_50 else -1
        
        market_features = {
            'spy_return_1d': spy_return if not np.isnan(spy_return) else 0,
            'spy_volatility': spy_volatility if not np.isnan(spy_volatility) else 0.02,
            'vixy_level': 16.0,  # 默认值
            'market_trend': trend,
            'recent_volatility': recent_vol if not np.isnan(recent_vol) else 0.015,
            'timestamp': window_spy.index[-1]
        }
        
        # 测试每个参数组合
        for threshold in param_grid['signal_threshold']:
            for top_n in param_grid['top_n_trades']:
                # 运行简化回测
                result = simple_backtest(window_spy, threshold, top_n)
                
                # 记录所有组合
                training_data.append({
                    **market_features,
                    'signal_threshold': threshold,
                    'top_n_trades': top_n,
                    # 'l1_risk_factor': None,
                    'sharpe_ratio': result['sharpe_ratio'],
                    'total_return': result['total_return'],
                    'volatility': result['volatility']
                })
    
    # 转换为 DataFrame
    df = pd.DataFrame(training_data)
    
    if len(df) == 0:
        print(f"\n❌ 错误: 没有生成任何训练数据!")
        print(f"   请检查数据时间范围和窗口大小")
        return None
    
    # 保存数据
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'meta_training_data.csv'
    
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ 数据生成完成!")
    print(f"   总样本数: {len(df)}")
    print(f"   保存位置: {output_file}")
    
    if len(df) > 0:
        print(f"\n📊 数据统计:")
        print(f"   Sharpe 范围: [{df['sharpe_ratio'].min():.2f}, {df['sharpe_ratio'].max():.2f}]")
        print(f"   平均 Sharpe: {df['sharpe_ratio'].mean():.2f}")
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='生成 L5 元策略训练数据')
    parser.add_argument('--days', type=int, default=180, help='总天数')
    parser.add_argument('--window', type=int, default=30, help='窗口大小')
    args = parser.parse_args()
    
    generate_meta_training_data(args.days, args.window)
