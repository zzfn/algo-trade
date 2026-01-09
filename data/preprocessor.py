import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from utils.logger import setup_logger

logger = setup_logger("preprocessor")

class DataPreprocessor:
    """
    数据预处理管道,包含:
    1. 时间戳对齐与填充
    2. 异常值去噪
    3. Log Returns 计算
    4. Z-Score 标准化
    """
    
    def __init__(self):
        self.scaler_stats: Optional[Dict[str, Dict[str, float]]] = None
        
    def clean_pipeline(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        完整的数据清洗流程
        
        Args:
            df: 原始 OHLCV 数据
            is_training: 是否为训练模式 (影响标准化行为)
            
        Returns:
            清洗后的数据
        """
        logger.info("🧹 开始数据预处理流程...")
        
        # 1. 时间戳对齐
        df = self.align_timestamps(df)
        logger.info(f"  ✓ 时间戳对齐完成: {len(df)} 行")
        
        # 2. 缺失值填充
        df = self.fill_missing(df)
        logger.info(f"  ✓ 缺失值填充完成: {len(df)} 行")
        
        # 3. 异常值去除
        df = self.remove_outliers(df)
        logger.info(f"  ✓ 异常值去除完成: {len(df)} 行")
        
        # 4. Log Returns (在特征工程之前计算基础收益率)
        df = self.add_log_returns(df)
        logger.info(f"  ✓ Log Returns 计算完成")
        
        logger.info("✅ 数据预处理完成!")
        return df
    
    def align_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        多标的时间戳对齐,确保所有标的在相同时间点都有数据
        
        策略:
        - 按 symbol 分组
        - 使用完整时间序列 reindex
        - 对齐后的缺失值将在 fill_missing 中处理
        """
        if 'symbol' not in df.columns:
            logger.warning("  ⚠️  数据中没有 symbol 列,跳过时间对齐")
            return df
        
        # 获取所有时间戳的并集
        all_timestamps = df['timestamp'].unique()
        all_timestamps = pd.Series(all_timestamps).sort_values().values
        
        aligned_groups = []
        for symbol, group in df.groupby('symbol'):
            # 创建完整时间索引
            group = group.set_index('timestamp').sort_index()
            group = group.reindex(all_timestamps)
            group['symbol'] = symbol
            aligned_groups.append(group.reset_index().rename(columns={'index': 'timestamp'}))
        
        df_aligned = pd.concat(aligned_groups, ignore_index=True)
        return df_aligned
    
    def fill_missing(self, df: pd.DataFrame, max_consecutive_na: int = 5) -> pd.DataFrame:
        """
        智能填充缺失值
        
        策略:
        - OHLCV: 前向填充 (ffill) - 假设价格在缺失期间保持不变
        - 其他列: 线性插值
        - 超过 max_consecutive_na 的连续缺失: 删除
        """
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        def fill_group(group):
            # OHLCV 前向填充
            for col in ohlcv_cols:
                if col in group.columns:
                    group[col] = group[col].ffill()
            
            # 其他数值列线性插值
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            other_cols = [c for c in numeric_cols if c not in ohlcv_cols]
            for col in other_cols:
                group[col] = group[col].interpolate(method='linear', limit=max_consecutive_na)
            
            return group
        
        if 'symbol' in df.columns:
            df = df.groupby('symbol', group_keys=False).apply(fill_group)
        else:
            df = fill_group(df)
        
        # 删除仍然存在的 NaN (超过最大连续缺失)
        initial_rows = len(df)
        df = df.dropna(subset=ohlcv_cols)
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.info(f"  ⚠️  删除了 {dropped} 行无法填充的数据")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        检测并移除异常值
        
        方法:
        1. IQR 方法: Q1 - 1.5*IQR, Q3 + 1.5*IQR
        2. 闪崩检测: 单周期跌幅 > 20%
        3. 错误报价: volume = 0 但价格变动
        """
        initial_rows = len(df)
        
        def detect_outliers(group):
            # 1. 闪崩检测 (单周期跌幅 > 20%)
            returns = group['close'].pct_change()
            flash_crash = returns < -0.20
            
            # 2. 错误报价 (volume = 0 但价格变动)
            price_change = group['close'].diff().abs() > 0
            zero_volume = group['volume'] == 0
            bad_quotes = price_change & zero_volume
            
            # 3. IQR 异常值检测 (针对 volume 和 returns)
            if method == 'iqr':
                # Volume 异常
                Q1_vol = group['volume'].quantile(0.25)
                Q3_vol = group['volume'].quantile(0.75)
                IQR_vol = Q3_vol - Q1_vol
                volume_outliers = (group['volume'] < Q1_vol - 1.5 * IQR_vol) | \
                                 (group['volume'] > Q3_vol + 1.5 * IQR_vol)
                
                # Returns 异常 (需要先计算)
                if len(returns.dropna()) > 0:
                    Q1_ret = returns.quantile(0.25)
                    Q3_ret = returns.quantile(0.75)
                    IQR_ret = Q3_ret - Q1_ret
                    return_outliers = (returns < Q1_ret - 3 * IQR_ret) | \
                                     (returns > Q3_ret + 3 * IQR_ret)
                else:
                    return_outliers = pd.Series(False, index=group.index)
            else:
                volume_outliers = pd.Series(False, index=group.index)
                return_outliers = pd.Series(False, index=group.index)
            
            # 组合所有异常标记
            is_outlier = flash_crash | bad_quotes | volume_outliers | return_outliers
            return group[~is_outlier]
        
        if 'symbol' in df.columns:
            df = df.groupby('symbol', group_keys=False).apply(detect_outliers)
        else:
            df = detect_outliers(df)
        
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info(f"  🗑️  移除了 {removed} 个异常值 ({removed/initial_rows:.2%})")
        
        return df.reset_index(drop=True)
    
    def add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Log Returns (对数收益率)
        
        优势:
        - 更好的统计特性 (近似正态分布)
        - 时间可加性: log_ret(t1->t3) = log_ret(t1->t2) + log_ret(t2->t3)
        - 对称性: ±10% 在 log 空间是对称的
        """
        def calc_log_returns(group):
            # 1 周期 Log Returns
            group['log_return_1p'] = np.log(group['close'] / group['close'].shift(1))
            
            # 5 周期 Log Returns
            group['log_return_5p'] = np.log(group['close'] / group['close'].shift(5))
            
            # 替换 inf 和 -inf (除以 0 的情况)
            group['log_return_1p'] = group['log_return_1p'].replace([np.inf, -np.inf], np.nan)
            group['log_return_5p'] = group['log_return_5p'].replace([np.inf, -np.inf], np.nan)
            
            return group
        
        if 'symbol' in df.columns:
            df = df.groupby('symbol', group_keys=False).apply(calc_log_returns)
        else:
            df = calc_log_returns(df)
        
        return df
    
    def standardize_features(self, df: pd.DataFrame, 
                            feature_cols: List[str],
                            fit: bool = True) -> pd.DataFrame:
        """
        Z-Score 标准化特征
        
        Args:
            df: 数据
            feature_cols: 需要标准化的特征列
            fit: 是否拟合统计量 (训练时 True, 预测时 False)
            
        Returns:
            标准化后的数据
        """
        # 排除不需要标准化的列
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        exclude_cols += [c for c in df.columns if c.startswith('target_')]
        
        cols_to_scale = [c for c in feature_cols if c not in exclude_cols and c in df.columns]
        
        if fit:
            # 训练模式: 计算并保存统计量
            self.scaler_stats = {}
            for col in cols_to_scale:
                mean = df[col].mean()
                std = df[col].std()
                self.scaler_stats[col] = {'mean': mean, 'std': std}
                
                # 标准化
                if std > 1e-8:  # 避免除以 0
                    df[col] = (df[col] - mean) / std
                else:
                    logger.warning(f"  ⚠️  列 {col} 标准差接近 0, 跳过标准化")
            
            logger.info(f"  📊 标准化了 {len(cols_to_scale)} 个特征")
        else:
            # 预测模式: 使用训练时的统计量
            if self.scaler_stats is None:
                raise ValueError("预测模式下必须先加载 scaler_stats (调用 load_scaler_stats)")
            
            for col in cols_to_scale:
                if col in self.scaler_stats:
                    mean = self.scaler_stats[col]['mean']
                    std = self.scaler_stats[col]['std']
                    if std > 1e-8:
                        df[col] = (df[col] - mean) / std
                else:
                    logger.warning(f"  ⚠️  列 {col} 没有保存的统计量, 跳过标准化")
        
        return df
    
    def save_scaler_stats(self, filepath: str):
        """保存标准化统计量"""
        if self.scaler_stats is None:
            raise ValueError("没有可保存的统计量")
        
        import joblib
        joblib.dump(self.scaler_stats, filepath)
        logger.info(f"💾 标准化统计量已保存到: {filepath}")
    
    def load_scaler_stats(self, filepath: str):
        """加载标准化统计量"""
        import joblib
        self.scaler_stats = joblib.load(filepath)
        logger.info(f"📦 标准化统计量已加载: {filepath}")


if __name__ == "__main__":
    # 测试代码
    from datetime import datetime, timedelta
    
    # 创建测试数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    symbols = ['AAPL', 'GOOGL']
    
    data = []
    for sym in symbols:
        for i, d in enumerate(dates):
            # 模拟一些缺失值和异常值
            if i % 20 == 0:  # 每 20 个周期缺失一次
                continue
            
            price = 100 + np.random.randn() * 5
            if i == 50:  # 模拟闪崩
                price = 50
            
            data.append({
                'timestamp': d,
                'symbol': sym,
                'open': price,
                'high': price + abs(np.random.randn()),
                'low': price - abs(np.random.randn()),
                'close': price,
                'volume': max(0, 1000 + np.random.randn() * 100)
            })
    
    df = pd.DataFrame(data)
    print(f"原始数据: {len(df)} 行")
    
    # 测试预处理
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_pipeline(df, is_training=True)
    print(f"\n清洗后数据: {len(df_clean)} 行")
    print(f"\nLog Returns 统计:")
    print(df_clean['log_return_1p'].describe())
