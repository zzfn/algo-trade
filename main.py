import os
import sys
from dotenv import load_dotenv
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from data.provider import DataProvider
from features.builder import FeatureBuilder
from models.trainer import QQQModelTrainer

# Load environment variables from .env file
load_dotenv()

def main():
    # 1. 默认配置
    symbol = "QQQ"
    timeframe = TimeFrame.Day 
    
    # 尝试从命令行参数获取周期 (如: python main.py 15m)
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '1d':
            timeframe = TimeFrame.Day
        elif arg == '1h':
            timeframe = TimeFrame.Hour
        elif arg.endswith('m'):
            try:
                mins = int(arg.replace('m', ''))
                timeframe = TimeFrame(mins, TimeFrameUnit.Minute)
            except ValueError:
                print(f"警告: 无法解析周期 '{arg}'，将使用默认日线(1d)。")
        else:
            print(f"警告: 未知的周期格式 '{arg}'，将使用默认日线(1d)。")
    
    # 转换周期为业界规范字符串 (如 1d, 15m)
    tf_str = DataProvider.get_tf_string(timeframe)
    
    # 根据周期自动调整获取数据的长度
    if timeframe.unit == TimeFrameUnit.Day:
        days_to_fetch = 365 * 5 # 日线看 5 年
    elif timeframe.unit == TimeFrameUnit.Hour:
        days_to_fetch = 365 * 1 # 小时线看 1 年
    else:
        days_to_fetch = 60      # 分钟线看两个月 (防止数据量过大)
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_fetch)
    
    print(f"正在启动 {symbol} ({tf_str}) 训练流水线...")
    
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
        model_filename = f"output/{symbol}_{tf_str}_lgbm.joblib"
        trainer.save_model(model_filename)
        
        print("流水线执行成功。")
        
    except Exception as e:
        print(f"流水线执行过程中出错: {e}")
        print("\n提示: 请确保已设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY。")

if __name__ == "__main__":
    main()
