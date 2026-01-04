import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Union
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

class DataProvider:
    @staticmethod
    def get_tf_string(tf: TimeFrame) -> str:
        """
        将 Alpaca TimeFrame 转换为业界通用字符串 (如 1d, 15m, 1h)
        """
        unit_map = {
            TimeFrameUnit.Minute: 'm',
            TimeFrameUnit.Hour: 'h',
            TimeFrameUnit.Day: 'd',
            TimeFrameUnit.Week: 'w',
            TimeFrameUnit.Month: 'M'
        }
        unit_str = unit_map.get(tf.unit, 'u')
        return f"{tf.amount}{unit_str}"

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Key and Secret Key must be provided or set as environment variables.")
        
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)

    def fetch_bars(self, symbols: Union[str, List[str]], timeframe: TimeFrame, start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch historical bars for one or multiple symbols.
        Returns a DataFrame with 'timestamp' and 'symbol' columns.
        """
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX
        )
        
        bars = self.client.get_stock_bars(request_params)
        df = bars.df
        
        # 处理多索引
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        else:
            # 如果是单标的，手动加上 symbol 列保持格式统一
            df = df.reset_index()
            if isinstance(symbols, str):
                df['symbol'] = symbols
        
        # 确保 timestamp 列是 tz-naive 的，方便后面处理
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            
        return df

if __name__ == "__main__":
    # Example usage (will fail if keys not set)
    try:
        provider = DataProvider()
        # Fetch last 30 days of daily data for QQQ
        end = datetime.now()
        start = end - timedelta(days=30)
        data = provider.fetch_bars("QQQ", TimeFrame.Day, start, end)
        print(data.head())
    except Exception as e:
        print(f"Error fetching data: {e}")
