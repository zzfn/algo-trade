import pandas as pd
import numpy as np

class FeatureBuilder:
    def __init__(self):
        pass

    def add_all_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        Supports multi-symbol DataFrames by grouping by 'symbol'.
        """
        df = df.copy()
        
        # 针对每个标的独立计算技术指标
        processed_groups = []
        for symbol, group in df.groupby('symbol'):
            # 排除 symbol 列传给处理函数，处理完后再加回
            processed = self._add_indicators_per_symbol(group.drop(columns='symbol', errors='ignore'))
            processed['symbol'] = symbol
            processed_groups.append(processed)
        df = pd.concat(processed_groups).reset_index(drop=True)
        
        # 添加全局特征 (如洗盘信号)
        df = self.add_shakeout_features(df)
        
        if is_training:
            # 添加信号标签 (Classification: 未来 5 周期是否能涨 > 0.5%)
            df = self.add_classification_target(df)
            df = df.dropna()
        else:
            # 预测模式下，只删除特征为 NaN 的行
            feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
            df = df.dropna(subset=feature_cols)
            
        return df

    def merge_l1_features(self, df: pd.DataFrame, l1_features: pd.DataFrame) -> pd.DataFrame:
        """
        将 L1 宏观特征合并到技术特征 DataFrame 中。
        
        Args:
            df: 技术特征 DataFrame (包含 timestamp, symbol 等)
            l1_features: L1 宏观特征 DataFrame (包含 timestamp 和宏观指标)
            
        Returns:
            合并后的 DataFrame,每行都包含对应时间的 L1 特征
        """
        # 确保 timestamp 列是 datetime 类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        l1_features['timestamp'] = pd.to_datetime(l1_features['timestamp'])
        
        # 只提取 L1 宏观特征列 (排除 OHLCV 和目标列)
        # L1 特征以 spy_, vixy_, tlt_ 开头
        l1_cols = [c for c in l1_features.columns 
                   if c.startswith(('spy_', 'vixy_', 'tlt_')) 
                   and c not in ['target_spy_5d']]
        
        if not l1_cols:
            print("  Warning: No L1 macro features found!")
            return df
        
        # 使用 merge_asof 进行时间对齐 (向前填充最近的 L1 数据)
        # 因为 L1 是日线,而 df 可能是分钟线或小时线
        df = df.sort_values('timestamp')
        l1_features = l1_features.sort_values('timestamp')
        
        # 对每个 symbol 分别合并 (虽然 L1 对所有 symbol 都一样)
        merged_groups = []
        for symbol, group in df.groupby('symbol'):
            merged = pd.merge_asof(
                group,
                l1_features[['timestamp'] + l1_cols],
                on='timestamp',
                direction='backward'  # 使用最近的历史 L1 数据
            )
            merged_groups.append(merged)
        
        df = pd.concat(merged_groups).reset_index(drop=True)
        
        return df



    def _add_indicators_per_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对单只标的计算所有指标
        """
        df = df.sort_values('timestamp')
        df['atr'] = self._calculate_atr(df) # 用于归一化
        
        df = self.add_returns(df)
        df = self.add_emas(df)
        df = self.add_adx(df)
        df = self.add_bollinger_bands(df)
        df = self.add_volume_features(df)
        df = self.add_volatility(df)
        df = self.add_price_action(df)
        df = self.add_smc_features(df)
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['return_1p'] = df['close'].pct_change(1)
        df['return_5p'] = df['close'].pct_change(5)
        df['target_return'] = df['close'].pct_change().shift(-1)
        return df

    def add_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_20_rel'] = (df['close'].ewm(span=20, adjust=False).mean() / df['close']) - 1
        return df

    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index (ADX).
        捕捉趋势强度，不分方向。
        """
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm[~((plus_dm > minus_dm) & (plus_dm > 0))] = 0
        minus_dm[~((minus_dm > plus_dm) & (minus_dm > 0))] = 0
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # Wilder's Smoothing
        atr = tr.rolling(window=period).mean() # 简化版 TR 均值作为平滑
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-9))
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        df['adx'] = dx.rolling(window=period).mean() / 100.0 # 归一化到 0-1
        return df

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['bb_upper_rel'] = (sma + (std * 2)) / df['close'] - 1
        df['bb_lower_rel'] = (sma - (std * 2)) / df['close'] - 1
        df['bb_width'] = (2 * 2 * std) / (sma + 1e-9)
        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_5'] + 1e-9)
        return df

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
        return df

    def add_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        # candle range 归一化
        atr = df['atr'] + 1e-9
        df['body_size_rel'] = (df['close'] - df['open']).abs() / atr
        df['candle_range_rel'] = (df['high'] - df['low']) / atr
        df['upper_wick_rel'] = (df['high'] - df[['open', 'close']].max(axis=1)) / atr
        df['lower_wick_rel'] = (df[['open', 'close']].min(axis=1) - df['low']) / atr
        df['wick_ratio'] = (df['upper_wick_rel'] + df['lower_wick_rel']) / (df['candle_range_rel'] + 1e-6)
        df['is_pin_bar'] = ((df['upper_wick_rel'] > df['body_size_rel'] * 2) | 
                            (df['lower_wick_rel'] > df['body_size_rel'] * 2)).astype(int)
        df['is_engulfing'] = ((df['body_size_rel'] > df['body_size_rel'].shift(1)) & 
                              (df['close'].pct_change() * df['close'].shift(1).pct_change() < 0)).astype(int)
        return df

    def add_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced Smart Money Concepts features.
        """
        atr = df['atr'] + 1e-9
        # 1. Swing Highs and Lows (Fractals) - 修复数据泄露
        # 只有在 t+window 时刻才能确认 t 时刻是高点/低点
        # 所以我们需要向后 shift(window) 来模拟实盘中的滞后确认
        window = 3
        
        # 原始逻辑(泄露): t时刻知道 t+3 的数据
        # is_high = df['high'] == df['high'].rolling(window=window*2+1, center=True).max()
        
        # 修复逻辑: 
        # t-3 时刻的高点,需要在 t 时刻才能确认
        # 所以我们计算 rolling max,然后判断中间那个点是否是最大值
        
        # 使用 rolling window 计算过去 2*window+1 根K线
        rolling_max = df['high'].rolling(window=window*2+1).max()
        rolling_min = df['low'].rolling(window=window*2+1).min()
        
        # 判断 window 根K线之前的那个点是否是局部极值
        # 注意: 这里确实会有 window 根K线的延迟,这是正常的
        df['swing_high'] = (df['high'].shift(window) == rolling_max).astype(int)
        df['swing_low'] = (df['low'].shift(window) == rolling_min).astype(int)
        
        # 2. Market Structure (BOS / CHoCH)
        # 使用 shift(1) 避免直接使用当前 bar 的 swing点 (虽然已经滞后了,但为了安全)
        last_h = df['high'].where(df['swing_high'] == 1).ffill()
        last_l = df['low'].where(df['swing_low'] == 1).ffill()
        
        # BOS: 收盘价突破前高/前低
        # 注意: 这里比较的是当前的 close 和 之前确认的 high
        df['bos_up'] = ((df['close'] > last_h.shift(1)) & (df['close'].shift(1) <= last_h.shift(1))).astype(int)
        df['bos_down'] = ((df['close'] < last_l.shift(1)) & (df['close'].shift(1) >= last_l.shift(1))).astype(int)
        
        # 3. Fair Value Gaps (FVG) - 归一化大小
        df['fvg_up'] = ((df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])).astype(int)
        df['fvg_down'] = ((df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])).astype(int)
        # 使用向量化操作替代 df.apply，避免因非连续索引导致的 KeyError
        df['fvg_size_rel'] = np.where(
            df['fvg_up'] == 1, df['low'] - df['high'].shift(2),
            np.where(df['fvg_down'] == 1, df['low'].shift(2) - df['high'], 0)
        ) / atr
        
        # 4. Displacement
        df['displacement'] = ( (df['high'] - df['low']) > atr * 1.5).astype(int)
        
        # 5. Order Blocks (OB)
        df['ob_bullish'] = ((df['displacement'] == 1) & (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1))).astype(int)
        df['ob_bearish'] = ((df['displacement'] == 1) & (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1))).astype(int)
        
        return df

    def add_classification_target(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.005) -> pd.DataFrame:
        """
        计算多分类信号标签。
        1 (Long): 未来 horizon 周期内最高价涨幅 > threshold。
        2 (Short): 未来 horizon 周期内最低价跌幅 < -threshold。
        0 (Neutral): 其他。
        如果同时满足，以先达到的为准（简化版：Long 优先）。
        """
        def get_directional_target(group):
            # 未来 N 个周期的滚动最高和最低收益
            f_max = group['close'].shift(-1).rolling(window=horizon, min_periods=1).max() / group['close'] - 1
            f_min = group['close'].shift(-1).rolling(window=horizon, min_periods=1).min() / group['close'] - 1
            
            target = np.zeros(len(group))
            # 逻辑：Long 1, Short 2, Neutral 0
            target[f_max > threshold] = 1
            target[f_min < -threshold] = 2
            # 特殊处理：如果两者都满足，这里简化为 1 (可以根据需求精细化)
            target[(f_max > threshold) & (f_min < -threshold)] = 1
            return pd.Series(target, index=group.index)

        df['target_signal'] = df.groupby('symbol', group_keys=False).apply(get_directional_target, include_groups=False).astype(int)
        return df

    def add_rank_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        计算截面排名标签。
        用于 L2 Ranking 模型。
        """
        # 计算未来 N 个周期的收益率作为排名依据
        df['target_return'] = df.groupby('symbol')['close'].shift(-horizon) / df['close'] - 1
        # 在每个时间切片内进行排名 (0 到 N-1)
        df['target_rank'] = df.groupby('timestamp')['target_return'].rank(method='first', ascending=True) - 1
        return df

    def add_return_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        计算未来收益率标签。
        用于 L4 收益预测模型。
        
        Args:
            df: 带有 OHLCV 的 DataFrame
            horizon: 预测的未来周期数
            
        Returns:
            添加了 target_future_return 列的 DataFrame
        """
        df['target_future_return'] = df.groupby('symbol')['close'].shift(-horizon) / df['close'] - 1
        return df

    def add_shakeout_features(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        检测洗盘 (Shakeout/Stop-Hunt) 信号。
        逻辑：价格突破前高/低后，在 N 根 K 线内反向拉回。
        """
        def detect_shakeout(group):
            # 最近的 Swing High / Low
            group['local_high'] = group['high'].rolling(lookback).max().shift(1)
            group['local_low'] = group['low'].rolling(lookback).min().shift(1)
            
            # 多头洗盘 (Bullish Shakeout): 低点破了 local_low，但收盘拉回到 local_low 之上
            group['shakeout_bull'] = ((group['low'] < group['local_low']) & (group['close'] > group['local_low'])).astype(int)
            
            # 空头洗盘 (Bearish Shakeout): 高点破了 local_high，但收盘跌回到 local_high 之下
            group['shakeout_bear'] = ((group['high'] > group['local_high']) & (group['close'] < group['local_high'])).astype(int)
            
            return group[['shakeout_bull', 'shakeout_bear', 'local_high', 'local_low']]

        shakeouts = df.groupby('symbol', group_keys=False).apply(detect_shakeout, include_groups=False)
        df = pd.concat([df, shakeouts], axis=1)
        return df

    def add_risk_targets(self, df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
        """
        基于 SMC 概念计算止盈止损目标 (用于 L4 风控模型训练)。
        
        参数:
            horizon: 未来观察周期数
            
        返回:
            添加了以下列的 DataFrame:
            - target_tp_long_pct: 做多止盈目标 (百分比)
            - target_sl_long_pct: 做多止损目标 (百分比,负值)
            - target_tp_short_pct: 做空止盈目标 (百分比,负值)
            - target_sl_short_pct: 做空止损目标 (百分比,正值)
        """
        def calculate_risk_targets(group):
            # 1. 识别最近的 Swing High/Low (支撑阻力位)
            nearest_resistance = group['high'].where(group['swing_high'] == 1).ffill()
            nearest_support = group['low'].where(group['swing_low'] == 1).ffill()
            
            # 2. 计算未来实际触及的最高/最低点
            future_high = group['high'].shift(-1).rolling(window=horizon, min_periods=1).max()
            future_low = group['low'].shift(-1).rolling(window=horizon, min_periods=1).min()
            
            current_price = group['close']
            
            # 3. 做多止盈目标: 未来最高点,但参考阻力位
            actual_gain = (future_high / current_price - 1)
            resistance_limit = (nearest_resistance * 1.5 / current_price - 1)
            group['target_tp_long_pct'] = np.minimum(actual_gain, resistance_limit)
            
            # 4. 做多止损目标: 未来最低点,参考支撑位
            actual_loss = (future_low / current_price - 1)
            support_limit = (nearest_support * 0.95 / current_price - 1)
            group['target_sl_long_pct'] = np.maximum(actual_loss, support_limit)
            
            # 5. 做空止盈目标: 未来最低点 (做空盈利 = 价格下跌)
            group['target_tp_short_pct'] = -actual_gain
            
            # 6. 做空止损目标: 未来最高点 (做空亏损 = 价格上涨)
            group['target_sl_short_pct'] = -actual_loss
            
            return group
        
        # 按 symbol 分组计算 (不使用 include_groups=False 以保留 symbol 列)
        df = df.groupby('symbol', group_keys=False).apply(calculate_risk_targets)
        
        return df

if __name__ == "__main__":
    # Test with multi-symbol dummy data
    dates = pd.date_range(start='2023-01-01', periods=20)
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    data = []
    for s in symbols:
        for d in dates:
            data.append({
                'timestamp': d,
                'symbol': s,
                'open': 100 + np.random.randn(),
                'high': 105 + np.random.randn(),
                'low': 95 + np.random.randn(),
                'close': 100 + np.random.randn(),
                'volume': 1000 + np.random.randint(0, 500)
            })
    df = pd.DataFrame(data)
    builder = FeatureBuilder()
    res = builder.add_all_features(df)
    print(f"Columns: {res.columns.tolist()}")
    if 'target_signal' in res.columns:
        print(f"Signal distribution: \n{res['target_signal'].value_counts()}")
    print(res.head(5))
