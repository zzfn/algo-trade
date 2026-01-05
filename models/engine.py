import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data.provider import DataProvider
from features.macro import L1FeatureBuilder
from features.technical import FeatureBuilder
from models.trainer import SklearnClassifierTrainer, RankingModelTrainer, SignalClassifierTrainer, RiskModelTrainer
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

class StrategyEngine:
    def __init__(self):
        self.provider = DataProvider()
        self.l1_builder = L1FeatureBuilder()
        self.l2_builder = FeatureBuilder()
        
        # Load models
        self.l1_model = SklearnClassifierTrainer().load("models/artifacts/l1_market_timing.joblib")
        self.l2_model = RankingModelTrainer().load("models/artifacts/l2_stock_selection.joblib")
        self.l3_model = SignalClassifierTrainer().load("models/artifacts/l3_execution.joblib")
        
        self.l4_trainer = RiskModelTrainer()
        self.l4_models = {
            "tp_long": self.l4_trainer.load("models/artifacts/l4_risk_tp_long.joblib", "tp_long"),
            "sl_long": self.l4_trainer.load("models/artifacts/l4_risk_sl_long.joblib", "sl_long"),
            "tp_short": self.l4_trainer.load("models/artifacts/l4_risk_tp_short.joblib", "tp_short"),
            "sl_short": self.l4_trainer.load("models/artifacts/l4_risk_sl_short.joblib", "sl_short")
        }
        
        self.l2_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'MU', 'AMD', 'ORCL', 'INTC']
        self.l1_symbols = ['SPY', 'VIXY', 'TLT']

    def analyze(self, target_dt: datetime):
        """
        Run the full 4-layer analysis for a given timestamp.
        Returns a dictionary with results from each layer.
        """
        results = {}
        
        # --- L1: Market Timing ---
        l1_start = target_dt - timedelta(days=300)
        df_l1_dict = {sym: self.provider.fetch_bars(sym, TimeFrame.Day, l1_start, target_dt + timedelta(days=1)) for sym in self.l1_symbols}
        df_l1_feats = self.l1_builder.build_l1_features(df_l1_dict)
        df_l1_feats = df_l1_feats[df_l1_feats['timestamp'] <= target_dt]
        
        if df_l1_feats.empty:
            results['l1_safe'] = False
            results['l1_prob'] = 0.0
        else:
            latest_l1 = df_l1_feats.iloc[-1:]
            l1_features = ['spy_return_1d', 'spy_dist_ma200', 'vixy_level', 'vixy_change_1d', 'tlt_return_5d']
            prob = self.l1_model.predict_proba(latest_l1[l1_features])[0][1]
            results['l1_safe'] = prob > 0.5
            results['l1_prob'] = prob

        # --- L2: Stock Selection ---
        l2_start = target_dt - timedelta(days=60)
        df_l2_raw = self.provider.fetch_bars(self.l2_symbols, TimeFrame.Hour, l2_start, target_dt + timedelta(days=1))
        df_l2_feats = self.l2_builder.add_all_features(df_l2_raw, is_training=False)
        l2_valid = df_l2_feats[df_l2_feats['timestamp'] <= target_dt]
        
        if l2_valid.empty:
            results['l2_ranked'] = pd.DataFrame()
            return results
            
        last_h_ts = l2_valid['timestamp'].max()
        l2_latest = l2_valid[l2_valid['timestamp'] == last_h_ts].copy()
        l2_exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                      'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                      'max_future_return', 'target_signal', 'local_high', 'local_low']
        l2_features = [c for c in l2_latest.columns if c not in l2_exclude]
        l2_latest['rank_score'] = self.l2_model.predict(l2_latest[l2_features])
        results['l2_ranked'] = l2_latest.sort_values('rank_score', ascending=False)
        results['l2_timestamp'] = last_h_ts

        # --- L3: Execution Signal ---
        all_l2_symbols = l2_latest['symbol'].tolist()
        l3_start = target_dt - timedelta(days=10)
        df_l3_raw = self.provider.fetch_bars(all_l2_symbols, TimeFrame(15, TimeFrameUnit.Minute), l3_start, target_dt + timedelta(days=1))
        df_l3_feats = self.l2_builder.add_all_features(df_l3_raw, is_training=False)
        l3_valid = df_l3_feats[df_l3_feats['timestamp'] <= target_dt]
        
        if l3_valid.empty:
            results['l3_signals'] = pd.DataFrame()
        else:
            last_15m_ts = l3_valid['timestamp'].max()
            l3_latest = l3_valid[l3_valid['timestamp'] == last_15m_ts].copy()
            l3_exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                          'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                          'max_future_return', 'target_signal', 'local_high', 'local_low']
            l3_features = [c for c in l3_latest.columns if c not in l3_exclude]
            probs = self.l3_model.predict_proba(l3_latest[l3_features])
            l3_latest['long_p'] = probs[:, 1]
            l3_latest['short_p'] = probs[:, 2]
            results['l3_signals'] = l3_latest
            results['l3_timestamp'] = last_15m_ts

        return results

    def get_risk_params(self, symbol: str, direction: str, l2_ranked: pd.DataFrame):
        """
        Calculate SL/TP for a specific symbol and direction.
        """
        feat_row = l2_ranked[l2_ranked['symbol'] == symbol]
        if feat_row.empty:
            return None
        
        l2_exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                      'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                      'max_future_return', 'target_signal', 'local_high', 'local_low', 'rank_score']
        l2_features = [c for c in feat_row.columns if c not in l2_exclude]
        
        if direction == "long":
            tp_pct = self.l4_models['tp_long'].predict(feat_row[l2_features])[0]
            sl_pct = self.l4_models['sl_long'].predict(feat_row[l2_features])[0]
        else:
            tp_pct = self.l4_models['tp_short'].predict(feat_row[l2_features])[0]
            sl_pct = self.l4_models['sl_short'].predict(feat_row[l2_features])[0]
            
        return {"tp_pct": tp_pct, "sl_pct": sl_pct}
