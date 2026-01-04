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
        df = df.groupby('symbol', group_keys=True).apply(self._add_indicators_per_symbol, include_groups=False).reset_index(level=0).reset_index(drop=True)
        
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

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        # 为 Ranking 准备：下一期的实际收益率
        df['target_return'] = df['close'].pct_change().shift(-1)
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_ratio'] = df['ma_5'] / df['ma_20']
        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma + 1e-9)
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
        df['body_size'] = (df['close'] - df['open']).abs()
        df['candle_range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['candle_range'] + 1e-6)
        df['is_pin_bar'] = ((df['upper_wick'] > df['body_size'] * 2) | 
                            (df['lower_wick'] > df['body_size'] * 2)).astype(int)
        df['is_engulfing'] = ((df['body_size'] > df['body_size'].shift(1)) & 
                              (df['close'].pct_change() * df['close'].shift(1).pct_change() < 0)).astype(int)
        return df

    def add_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['fvg_up'] = ((df['low'] > df['high'].shift(2))).astype(int)
        df['fvg_down'] = ((df['high'] < df['low'].shift(2))).astype(int)
        df['atr_sim'] = df['candle_range'].rolling(window=20).mean()
        df['displacement'] = (df['candle_range'] > df['atr_sim'] * 2).astype(int)
        return df

    def add_rank_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算截面排名标签。
        在每一个 timestamp，根据 target_return 对不同 symbol 进行排序。
        收益率最高的标的分数最高。
        """
        # 注意：这里需要对整个 df 按 timestamp 分组进行排名
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
