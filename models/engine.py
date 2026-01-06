import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from data.provider import DataProvider
from features.macro import L1FeatureBuilder
from features.technical import FeatureBuilder
from models.trainer import SklearnClassifierTrainer, RankingModelTrainer, SignalClassifierTrainer, RiskModelTrainer
from models.smc_rules import get_smc_risk_params
from models.constants import (
    get_feature_columns, get_allocation_by_return,
    L1_SAFE_THRESHOLD, SIGNAL_THRESHOLD, L1_RISK_FACTOR,
    L1_LOOKBACK_DAYS, L2_LOOKBACK_DAYS, L3_LOOKBACK_DAYS,
    L1_SYMBOLS, L2_SYMBOLS, TOP_N_TRADES
)
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
        
        # L4: 收益预测模型 (新版)
        self.l4_trainer = RiskModelTrainer()
        self.l4_return_model = self.l4_trainer.load("models/artifacts/l4_return_predictor.joblib", "return_predictor")
        
        # 使用配置文件中的标的池
        self.l2_symbols = L2_SYMBOLS
        self.l1_symbols = L1_SYMBOLS

    def analyze(self, target_dt: datetime):
        """
        Run the full 4-layer analysis for a given timestamp.
        Returns a dictionary with results from each layer.
        """
        results = {}
        
        # --- L1: Market Timing ---
        l1_start = target_dt - timedelta(days=L1_LOOKBACK_DAYS)
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
            results['l1_safe'] = prob > L1_SAFE_THRESHOLD
            results['l1_prob'] = prob

        # --- L2: Stock Selection (15min) ---
        l2_start = target_dt - timedelta(days=L2_LOOKBACK_DAYS)
        df_l2_raw = self.provider.fetch_bars(self.l2_symbols, TimeFrame(15, TimeFrameUnit.Minute), l2_start, target_dt + timedelta(days=1))
        df_l2_feats = self.l2_builder.add_all_features(df_l2_raw, is_training=False)
        
        l2_valid = df_l2_feats[df_l2_feats['timestamp'] <= target_dt]
        
        if l2_valid.empty:
            results['l2_ranked'] = pd.DataFrame()
            return results
            
        last_h_ts = l2_valid['timestamp'].max()
        l2_latest = l2_valid[l2_valid['timestamp'] == last_h_ts].copy()
        
        # L2 模型预测 (不使用 L1 特征)
        l2_features = get_feature_columns(l2_latest)
        l2_latest['rank_score'] = self.l2_model.predict(l2_latest[l2_features])
        
        # 预测后合并 L1 特征 (供 L4 使用)
        l2_latest = self.l2_builder.merge_l1_features(l2_latest, df_l1_feats)
        
        results['l2_ranked'] = l2_latest.sort_values('rank_score', ascending=False)
        results['l2_timestamp'] = last_h_ts

        # --- L3: Trend Confirmation ---
        all_l2_symbols = l2_latest['symbol'].tolist()
        l3_start = target_dt - timedelta(days=L3_LOOKBACK_DAYS)
        df_l3_raw = self.provider.fetch_bars(all_l2_symbols, TimeFrame.Minute, l3_start, target_dt + timedelta(days=1))
        df_l3_feats = self.l2_builder.add_all_features(df_l3_raw, is_training=False)
        
        l3_valid = df_l3_feats[df_l3_feats['timestamp'] <= target_dt]
        
        if l3_valid.empty:
            results['l3_signals'] = pd.DataFrame()
        else:
            last_ts = l3_valid['timestamp'].max()
            l3_latest = l3_valid[l3_valid['timestamp'] == last_ts].copy()
            
            # L3 模型预测 (不使用 L1 特征)
            l3_features = get_feature_columns(l3_latest)
            probs = self.l3_model.predict_proba(l3_latest[l3_features])
            l3_latest['long_p'] = probs[:, 1]
            l3_latest['short_p'] = probs[:, 2]
            
            # 预测后合并 L1 特征 (供 L4 使用)
            l3_latest = self.l2_builder.merge_l1_features(l3_latest, df_l1_feats)
            
            results['l3_signals'] = l3_latest
            results['l3_timestamp'] = last_ts

        return results

    def predict_return(self, symbol: str, l2_ranked: pd.DataFrame) -> float:
        """
        预测指定标的的未来收益率 (L4)。
        
        Args:
            symbol: 标的代码
            l2_ranked: L2 排序后的 DataFrame (包含特征)
            
        Returns:
            预期收益率 (如 0.02 表示 2%)
        """
        feat_row = l2_ranked[l2_ranked['symbol'] == symbol]
        if feat_row.empty:
            return 0.0
        
        l2_features = get_feature_columns(feat_row)
        predicted_return = self.l4_return_model.predict(feat_row[l2_features])[0]
        return predicted_return

    def get_allocation(self, symbol: str, l2_ranked: pd.DataFrame, l1_safe: bool = True) -> float:
        """
        根据预期收益和市场环境计算仓位分配比例。
        
        Args:
            symbol: 标的代码
            l2_ranked: L2 排序后的 DataFrame
            l1_safe: L1 市场安全标志 (用于调整仓位)
            
        Returns:
            仓位分配比例 (如 0.10 表示 10%)
        """
        predicted_return = self.predict_return(symbol, l2_ranked)
        base_allocation = get_allocation_by_return(predicted_return)
        
        # L1 风险调整: 市场不安全时降低仓位
        if not l1_safe:
            # 使用配置的风险系数降低仓位
            return base_allocation * L1_RISK_FACTOR
        
        return base_allocation


    def get_risk_params(self, symbol: str, direction: str, l2_ranked: pd.DataFrame):
        """
        使用 SMC 规则引擎计算止盈止损。
        
        Args:
            symbol: 标的代码
            direction: 'long' 或 'short'
            l2_ranked: L2 排序后的 DataFrame
            
        Returns:
            dict: 包含 entry, stop_loss, take_profit, sl_pct, tp_pct 的字典
        """
        feat_row = l2_ranked[l2_ranked['symbol'] == symbol]
        if feat_row.empty:
            return None
        
        # 使用 SMC 规则引擎计算止盈止损
        row = feat_row.iloc[0]
        return get_smc_risk_params(row, direction)

    def filter_signals(self, l3_signals, direction="long", top_n=None):
        """
        过滤和排序信号，返回达到阈值的 top_n 个高置信度标的。
        
        Args:
            l3_signals: L3 趋势信号 DataFrame (包含 long_p, short_p 等列)
            direction: 'long' 或 'short'
            top_n: 返回的标的数量，默认使用 TOP_N_TRADES
            
        Returns:
            DataFrame: 过滤后的高置信度信号
        """
        if top_n is None:
            top_n = TOP_N_TRADES
            
        prob_col = 'long_p' if direction == 'long' else 'short_p'
        
        # 排序并取 top_n
        sorted_signals = l3_signals.sort_values(prob_col, ascending=False).head(top_n)
        # 过滤达到阈值的信号
        filtered = sorted_signals[sorted_signals[prob_col] > SIGNAL_THRESHOLD]
        
        return filtered

