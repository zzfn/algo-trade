import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame
from data.provider import DataProvider
from features.builder import FeatureBuilder
from models.trainer import QQQModelTrainer

# Load environment variables from .env file
load_dotenv()

def main():
    # 1. Setup
    symbol = "QQQ"
    timeframe = TimeFrame.Day
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)  # 5 years of data
    
    print(f"正在启动 {symbol} 训练流水线...")
    
    try:
        # 2. 获取数据
        provider = DataProvider()
        df = provider.fetch_bars(symbol, timeframe, start_date, end_date)
        print(f"成功获取 {len(df)} 条 K 线数据。")
        
        # 3. 特征工程
        builder = FeatureBuilder()
        df_features = builder.add_all_features(df)
        print(f"特征生成完毕。数据形状: {df_features.shape}")
        
        # 4. Define features and target
        feature_cols = [
            'return_1d', 'return_5d', 'ma_5', 'ma_20', 
            'ma_ratio', 'rsi', 'volatility_20d',
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'volume_ratio', 'volume_change'
        ]
        target_col = 'target'
        
        # 5. Train Model
        trainer = QQQModelTrainer()
        trainer.train(df_features, feature_cols, target_col)
        
        # 6. 保存模型
        os.makedirs("output", exist_ok=True)
        trainer.save_model("output/qqq_lgbm_model.joblib")
        
        print("流水线执行成功。")
        
    except Exception as e:
        print(f"流水线执行过程中出错: {e}")
        print("\n提示: 请确保已设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY。")

if __name__ == "__main__":
    main()
