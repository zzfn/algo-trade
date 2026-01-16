import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.provider import DataProvider
from features.macro import MacroFeatureBuilder
from features.technical import FeatureBuilder
from models.trainer import SklearnClassifierTrainer, RankingModelTrainer, SignalClassifierTrainer, RiskModelTrainer
from models.smc_rules import get_smc_risk_params
from models.constants import (
    get_feature_columns, get_allocation_by_return,
    SIGNAL_THRESHOLD,
    L1_LOOKBACK_DAYS, L2_LOOKBACK_DAYS, L3_LOOKBACK_DAYS,
    MACRO_SYMBOLS, L2_SYMBOLS, TOP_N_TRADES
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from utils.logger import setup_logger

logger = setup_logger("engine")

class StrategyEngine:
    def __init__(self):
        self.provider = DataProvider()
        self.macro_builder = MacroFeatureBuilder()
        self.l2_builder = FeatureBuilder()
        
        # Load models
        # L1 Model Removed (Unified into L4)
        
        self.l2_model = RankingModelTrainer().load("models/artifacts/l2_stock_selection.joblib")
        # L3 might need update if it depends on 5min, but for now we focus on L1/L4 unification.
        
        self.l3_model = SignalClassifierTrainer().load("models/artifacts/l3_execution.joblib")
        
        # L4: Unified Return Predictor (Replacing separate L4)
        self.l4_trainer = RiskModelTrainer()
        self.l4_return_model = self.l4_trainer.load("models/artifacts/unified_return_predictor.joblib", "return_predictor")
        
        # 使用配置文件中的标的池
        self.l2_symbols = L2_SYMBOLS
        self.macro_symbols = MACRO_SYMBOLS

    def analyze(self, target_dt: datetime):
        """
        Run the full 4-layer analysis for a given timestamp.
        Returns a dictionary with results from each layer.
        """
        results = {}
        
        # --- Macro: Market Timing (Data Only) ---
        macro_start = target_dt - timedelta(days=L1_LOOKBACK_DAYS)
        
        # ✅ 批量获取所有市场指标数据 (1H)
        df_macro_all = self.provider.fetch_bars(
            self.macro_symbols,  # ['SPY', 'VIXY', 'TLT']
            TimeFrame.Hour,   # Unified to 1H
            macro_start, 
            target_dt + timedelta(days=1), 
            use_redis=True
        )
        
        # 按标的分组
        df_macro_dict = {}
        if not df_macro_all.empty:
            grouped = df_macro_all.groupby('symbol')
            for sym, df in grouped:
                df_macro_dict[sym] = df
        
        df_macro_feats = self.macro_builder.build_macro_features(df_macro_dict)
        df_macro_feats = df_macro_feats[df_macro_feats['timestamp'] <= target_dt]
        
        # Unified Model doesn't output a separate 'safe' flag.
        # We set it to True as the Unified Risk Model will handle bad markets by predicting low returns.
        results['macro_safe'] = True 
        results['macro_prob'] = 1.0
        
        if df_macro_feats.empty:
            logger.warning("Macro features empty!")

        # --- L2: Stock Selection (5min) ---
        # --- L2: Stock Selection (1H) ---
        l2_start = target_dt - timedelta(days=L2_LOOKBACK_DAYS)
        # Fetch 1H data
        df_l2_raw = self.provider.fetch_bars(self.l2_symbols, TimeFrame.Hour, l2_start, target_dt + timedelta(days=1), use_redis=True)
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
        l2_latest = self.l2_builder.merge_l1_features(l2_latest, df_macro_feats)
        
        results['l2_ranked'] = l2_latest.sort_values('rank_score', ascending=False)
        results['l2_timestamp'] = last_h_ts

        # --- L3: Trend Confirmation (1H) ---
        all_l2_symbols = l2_latest['symbol'].tolist()
        l3_start = target_dt - timedelta(days=L3_LOOKBACK_DAYS)
        # Fetch 1H data
        df_l3_raw = self.provider.fetch_bars(all_l2_symbols, TimeFrame.Hour, l3_start, target_dt + timedelta(days=1), use_redis=True)
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
            l3_latest = self.l2_builder.merge_l1_features(l3_latest, df_macro_feats)
            
            results['l3_signals'] = l3_latest
            results['l3_timestamp'] = last_ts
            
            # --- Freshness Check ---
            # 如果最新的 L3 数据 (1分钟线) 滞后超过 15 分钟 (允许少量延迟), 则认为是过期数据
            time_lag = target_dt - last_ts
            if time_lag > timedelta(minutes=15):
                logger.warning(f"⚠️  Data Stale Warning: Latest data is from {last_ts}, lag is {time_lag}. Skipping trading.")
                results['l3_signals'] = pd.DataFrame() # Clear signals to prevent trading
            elif time_lag > timedelta(minutes=5):
                 logger.warning(f"⚠️  Data Lag Warning: Latest data is from {last_ts}, lag is {time_lag}.")

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
        # Use absolute value to size position based on magnitude of opportunity
        base_allocation = get_allocation_by_return(abs(predicted_return))
        
        # Unified Model handles macro risk implicitly via 'predicted_return'.
        # If macro is bad, predicted_return should be low/negative.
        # We can still enforce a manual override if 'l1_safe' was passed as False, 
        # but in this architecture l1_safe is always True (or unused).
        
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

    def filter_signals(self, l3_signals, direction="long", top_n=None, threshold=None):
        """
        过滤和排序信号，返回达到阈值的 top_n 个高置信度标的。
        
        Args:
            l3_signals: L3 趋势信号 DataFrame (包含 long_p, short_p 等列)
            direction: 'long' 或 'short'
            top_n: 返回的标的数量，默认使用 TOP_N_TRADES
            threshold: 信号置信度阈值，默认使用 SIGNAL_THRESHOLD
            
        Returns:
            DataFrame: 过滤后的高置信度信号
        """
        if top_n is None:
            top_n = TOP_N_TRADES
            
        if threshold is None:
            threshold = SIGNAL_THRESHOLD
            
        prob_col = 'long_p' if direction == 'long' else 'short_p'
        
        # 排序并取 top_n
        sorted_signals = l3_signals.sort_values(prob_col, ascending=False).head(top_n)
        # 过滤达到阈值的信号
        filtered = sorted_signals[sorted_signals[prob_col] > threshold]
        
        return filtered

