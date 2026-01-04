.PHONY: train-l1 train-l2 train-l3 backtest predict help setup clean

# 默认目标
help:
	@echo "可用命令:"
	@echo "  make train-l1      - 训练 L1 择时模型 (Macro)"
	@echo "  make train-l2      - 训练 L2 选股模型 (Ranker)"
	@echo "  make train-l3      - 训练 L3 执行模型 (Signal)"
	@echo "  make predict       - 运行三层架构层级预测 (Real-time)"
	@echo "  make backtest tf=1h - 运行回测脚本"
	@echo "  make setup         - 安装依赖"
	@echo "  make clean         - 清理输出文件"

# 训练命令
train-l1:
	PYTHONPATH=. uv run python scripts/train_l1.py

train-l2:
	PYTHONPATH=. uv run python scripts/train_l2.py

train-l3:
	PYTHONPATH=. uv run python scripts/train_l3.py

# 预测命令
predict:
	PYTHONPATH=. uv run python predict.py

# 回测命令 (保留通用回测脚本支持)
backtest:
	PYTHONPATH=. uv run python scripts/backtest.py $(if $(tf),$(tf),1h) --days $(if $(days),$(days),90) --top_n 1 --model $(if $(model),$(model),models/artifacts/l2_stock_selection.joblib)

# 环境与清理
setup:
	uv sync

clean:
	rm -rf models/artifacts/*.joblib
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf scripts/legacy/__pycache__
