import pandas as pd
import numpy as np

class FeatureBuilder:
    def __init__(self):
        pass

    def add_all_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        """
        df = df.copy()
        df = self.add_returns(df)
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_volume_features(df)
        df = self.add_volatility(df)
        
        if is_training:
            df = self.add_target(df)
            # Drop rows with NaN values created by rolling windows and target shift
            df = df.dropna()
        else:
            # For prediction, we only drop rows where features are NaN (initial rolling window)
            # but keep the very last row even if we don't know the future target
            feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.dropna(subset=feature_cols)
            
        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
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
        
        rs = gain / loss
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
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        return df

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
        return df

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target: Predict the sign of the next day's return.
        1 for Up, 0 for Down.
        """
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        return df

if __name__ == "__main__":
    # Example usage with dummy data
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=100),
        'close': np.random.randn(100).cumsum() + 100,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 100,
        'low': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 5000, 100)
    }
    df = pd.DataFrame(data)
    builder = FeatureBuilder()
    df_with_features = builder.add_all_features(df)
    print(df_with_features.head())
    print(df_with_features.columns)
