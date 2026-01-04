import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

class DataProvider:
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Key and Secret Key must be provided or set as environment variables.")
        
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)

    def fetch_bars(self, symbol: str, timeframe: TimeFrame, start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch historical bars for a given symbol.
        """
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX
        )
        
        bars = self.client.get_stock_bars(request_params)
        df = bars.df
        
        # Reset index to have 'timestamp' as a column and remove 'symbol' if it's there
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            if 'symbol' in df.columns:
                df = df.drop(columns=['symbol'])
        
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
