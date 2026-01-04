import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.timeframe import TimeFrame
from data.provider import DataProvider
from features.technical import FeatureBuilder
from models.trainer import SignalClassifierTrainer

def main():
    # Load environment variables
    load_dotenv()
    
    # 1. Initialize components
    provider = DataProvider()
    builder = FeatureBuilder()
    trainer = SignalClassifierTrainer(model_name="Universal_PA_SMC_Classifier")

    # 2. Define universe and timeframe
    symbols = ["QQQ", "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "NFLX", "INTC"]
    tf = TimeFrame.Hour
    
    # Use last 1 year of data for a "universal" model
    end = datetime.now()
    start = end - timedelta(days=365)
    
    print(f"Fetching data for {len(symbols)} symbols from {start.date()} to {end.date()}...")
    
    # 3. Fetch data
    try:
        raw_df = provider.fetch_bars(symbols, tf, start, end)
        print(f"Total raw rows: {len(raw_df)}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # 4. Build features
    print("Building enhanced PA/SMC features...")
    df_with_features = builder.add_all_features(raw_df, is_training=True)
    print(f"Final training rows: {len(df_with_features)}")

    # 5. Define feature columns
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                    'target_return', 'target_rank', 'atr', 'vwap', 'trade_count', 
                    'max_future_return', 'target_signal']
    feature_cols = [c for c in df_with_features.columns if c not in exclude_cols]
    
    print(f"Training with {len(feature_cols)} features: {feature_cols}")

    # 6. Train model
    trainer.train(df_with_features, feature_cols, 'target_signal')

    # 7. Save model
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "universal_pa_smc_classifier.joblib")
    trainer.save_model(model_path)

if __name__ == "__main__":
    main()
