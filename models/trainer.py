import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class QQQModelTrainer:
    def __init__(self, model_params=None):
        self.model_params = model_params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 7,             # Reduced for small dataset
            'learning_rate': 0.01,       # Lower LR for better convergence
            'feature_fraction': 0.8,
            'min_child_samples': 20,
            'verbosity': -1              # Reduce noise
        }
        self.model = None

    def train(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        """
        Train the LightGBM model.
        """
        X = df[feature_cols]
        y = df[target_col]

        # 时间序列切分 (目前使用简单切分，实际交易中建议使用 Walk-Forward)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        print(f"训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}...")

        self.model = lgb.LGBMClassifier(**self.model_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='binary_logloss',
            # early_stopping_rounds=10 # Deprecated in newer versions, use callback if needed
        )

        # 评估
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"准确率 (Accuracy): {acc:.4f}")
        print("\n" + "="*50)
        print(f"{'类别':<10} | {'精确率':<10} | {'召回率':<10} | {'F1分数':<10} | {'样本量':<10}")
        print("-" * 50)
        
        # 映射类别名称
        label_map = {'0': '下跌 (0)', '1': '上涨 (1)'}
        
        for label, metrics in report_dict.items():
            if label in label_map:
                name = label_map[label]
                print(f"{name:<10} | {metrics['precision']:<12.4f} | {metrics['recall']:<12.4f} | {metrics['f1-score']:<12.4f} | {metrics['support']:<10}")
            elif label == 'accuracy':
                pass # 已打印
            elif label == 'macro avg':
                print("-" * 50)
                print(f"{'宏平均':<10} | {metrics['precision']:<12.4f} | {metrics['recall']:<12.4f} | {metrics['f1-score']:<12.4f} | {metrics['support']:<10}")
            elif label == 'weighted avg':
                print(f"{'加权平均':<10} | {metrics['precision']:<12.4f} | {metrics['recall']:<12.4f} | {metrics['f1-score']:<12.4f} | {metrics['support']:<10}")
        print("="*50 + "\n")

        return acc

    def save_model(self, filepath: str):
        if self.model:
            import joblib
            joblib.dump(self.model, filepath)
            print(f"模型已保存至 {filepath}")
        else:
            print("没有可保存的模型。")

if __name__ == "__main__":
    # Example usage with dummy data
    import numpy as np
    data = np.random.rand(100, 5)
    target = np.random.randint(0, 2, 100)
    df = pd.DataFrame(data, columns=[f'feat_{i}' for i in range(5)])
    df['target'] = target
    
    trainer = QQQModelTrainer()
    trainer.train(df, [f'feat_{i}' for i in range(5)], 'target')
