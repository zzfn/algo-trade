import pandas as pd
import numpy as np

class L1FeatureBuilder:
    def build_l1_features(self, df_dict: dict) -> pd.DataFrame:
        """
        df_dict should contain DataFrames for 'SPY', 'VIXY', 'TLT'.
        All should have daily bars with 'timestamp', 'close', etc.
        """
        spy = df_dict['SPY'].copy().set_index('timestamp').sort_index()
        vixy = df_dict['VIXY'].copy().set_index('timestamp').sort_index()
        tlt = df_dict['TLT'].copy().set_index('timestamp').sort_index()
        
        # 1. SPY Features
        spy['spy_return_1d'] = spy['close'].pct_change(1)
        spy['spy_ma_20'] = spy['close'].rolling(20).mean()
        spy['spy_ma_200'] = spy['close'].rolling(200).mean()
        spy['spy_dist_ma200'] = spy['close'] / spy['spy_ma_200'] - 1
        
        # 2. VIXY Features (Volatility)
        vixy['vixy_level'] = vixy['close']
        vixy['vixy_change_1d'] = vixy['close'].pct_change(1)
        
        # 3. TLT Features (Rates Proxy)
        tlt['tlt_return_5d'] = tlt['close'].pct_change(5)
        
        # 合并数据
        df = spy.join(vixy[['vixy_level', 'vixy_change_1d']], how='inner')
        df = df.join(tlt[['tlt_return_5d']], how='inner')
        
        # 4. Target: SPY 5d forward return > 0
        df['target_spy_5d'] = (df['close'].shift(-5) > df['close']).astype(int)
        
        # 清理
        df = df.dropna()
        return df.reset_index()

if __name__ == "__main__":
    print("L1FeatureBuilder ready.")
