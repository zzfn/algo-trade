"""
训练 L5 元策略模型
"""

import pandas as pd
from strategies.meta_strategy import MetaStrategyModel
from pathlib import Path

def train_l5_model():
    """训练 L5 元策略模型"""
    
    print("=" * 60)
    print("🧠 训练 L5 元策略模型")
    print("=" * 60)
    
    # 加载训练数据
    data_file = Path('data/meta_training_data.csv')
    
    if not data_file.exists():
        print(f"❌ 错误: 训练数据不存在!")
        print(f"   请先运行: make generate-meta-data")
        return
    
    print(f"\n📊 加载训练数据...")
    df = pd.read_csv(data_file)
    print(f"   样本数: {len(df)}")
    print(f"   特征列: {df.columns.tolist()}")
    
    # 数据统计
    print(f"\n📈 数据统计:")
    print(f"   Sharpe 范围: [{df['sharpe_ratio'].min():.2f}, {df['sharpe_ratio'].max():.2f}]")
    print(f"   平均 Sharpe: {df['sharpe_ratio'].mean():.2f}")
    
    # 训练模型
    print(f"\n🔧 开始训练...")
    model = MetaStrategyModel()
    model.train(df)
    
    # 保存模型
    output_path = 'models/artifacts/l5_meta_strategy.joblib'
    model.save(output_path)
    
    # 测试预测
    print(f"\n🧪 测试预测...")
    test_features = {
        'spy_return_1d': 0.01,
        'spy_volatility': 0.02,
        'vixy_level': 16.0,
        'market_trend': 1,
        'recent_volatility': 0.015
    }
    
    optimal_params = model.predict_optimal_params(test_features)
    print(f"   测试市场特征: {test_features}")
    print(f"   预测最优参数: {optimal_params}")
    
    print(f"\n✅ L5 模型训练完成!")

if __name__ == "__main__":
    train_l5_model()
