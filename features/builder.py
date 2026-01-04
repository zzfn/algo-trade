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
        
        if is_training:
            # 添加截面排名标签 (Cross-sectional Ranking)
            df = self.add_rank_target(df)
            df = df.dropna()
        else:
            # 预测模式下，只删除特征为 NaN 的行
            feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
            df = df.dropna(subset=feature_cols)
            
        return df

    def _add_indicators_per_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对单只标的计算所有指标
        """
        df = df.sort_values('timestamp')
        df['atr'] = self._calculate_atr(df) # 用于归一化
        
        df = self.add_returns(df)
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
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
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['target_return'] = df['close'].pct_change().shift(-1)
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ma_5_rel'] = (df['close'].rolling(window=5).mean() / df['close']) - 1
        df['ma_20_rel'] = (df['close'].rolling(window=20).mean() / df['close']) - 1
        df['ma_ratio'] = (df['ma_5_rel'] + 1) / (df['ma_20_rel'] + 1)
        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = (100 - (100 / (1 + rs))) / 100 # 归一化到 0-1
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        # MACD 归一化为价格的百分比
        df['macd_rel'] = (exp1 - exp2) / df['close']
        df['macd_signal_rel'] = df['macd_rel'].ewm(span=9, adjust=False).mean()
        df['macd_hist_rel'] = df['macd_rel'] - df['macd_signal_rel']
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
        # 1. Swing Highs and Lows (Fractals)
        window = 3
        df['swing_high'] = (df['high'] == df['high'].rolling(window=window*2+1, center=True).max()).astype(int)
        df['swing_low'] = (df['low'] == df['low'].rolling(window=window*2+1, center=True).min()).astype(int)
        
        # 2. Market Structure (BOS / CHoCH)
        last_h = df['high'].where(df['swing_high'] == 1).ffill()
        last_l = df['low'].where(df['swing_low'] == 1).ffill()
        
        df['bos_up'] = ((df['close'] > last_h.shift(1)) & (df['high'] != last_h)).astype(int)
        df['bos_down'] = ((df['close'] < last_l.shift(1)) & (df['low'] != last_l)).astype(int)
        
        # 3. Fair Value Gaps (FVG) - 归一化大小
        df['fvg_up'] = ((df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])).astype(int)
        df['fvg_down'] = ((df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])).astype(int)
        df['fvg_size_rel'] = df.apply(lambda x: (x['low'] - df.loc[x.name-2, 'high']) if x['fvg_up'] else 
                                     (df.loc[x.name-2, 'low'] - x['high']) if x['fvg_down'] else 0, axis=1) / atr
        
        # 4. Displacement
        df['displacement'] = ( (df['high'] - df['low']) > atr * 1.5).astype(int)
        
        # 5. Order Blocks (OB)
        df['ob_bullish'] = ((df['displacement'] == 1) & (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1))).astype(int)
        df['ob_bearish'] = ((df['displacement'] == 1) & (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1))).astype(int)
        
        return df

    def add_rank_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算截面排名标签。
        """
        df['target_rank'] = df.groupby('timestamp')['target_return'].rank(method='first', ascending=True) - 1
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
    print(res[['timestamp', 'symbol', 'target_return', 'target_rank']].head(15))
