.PHONY: train-l1 train-l2 train-l3 train-l4 backtest predict trade help setup clean

# 默认目标
help:
	@echo "可用命令:"
	@echo "  make train-l1      - 训练 L1 择时模型 (Macro)"
	@echo "  make train-l2      - 训练 L2 选股模型 (Ranker)"
	@echo "  make train-l3      - 训练 L3 趋势模型 (Trend)"
	@echo "  make train-l4      - 训练 L4 风控模型 (Risk Management)"
	@echo "  make predict       - 运行四层架构层级预测 (Real-time)"
	@echo "  make backtest-l1   - 回测 L1 (Macro / Market Timing)"
	@echo "  make backtest-l2   - 回测 L2 (Stock Selection)"
	@echo "  make backtest-l3   - 回测 L3 (Trend Confirmation)"
	@echo "  make backtest-l4   - 回测 L4 (Risk & Allocation)"
	@echo "  make backtest tf=1h - 运行多空策略回测"
	@echo "  make trade         - 运行全自动交易"
	@echo "  make setup         - 安装依赖"
	@echo "  make clean         - 清理输出文件"

# 训练命令
train-l1:
	PYTHONPATH=. uv run python scripts/train_l1.py

train-l2:
	PYTHONPATH=. uv run python scripts/train_l2.py

train-l3:
	PYTHONPATH=. uv run python scripts/train_l3.py

train-l4:
	PYTHONPATH=. uv run python scripts/train_l4.py

# 预测命令
predict:
	PYTHONPATH=. uv run python predict.py $(args)

# 全自动交易命令
trade:
	PYTHONPATH=. uv run python trade.py $(args)

# 回测命令 (保留通用回测脚本支持)
# 分层回测命令
backtest-l1:
	PYTHONPATH=. uv run python scripts/backtest_l1.py --days $(if $(days),$(days),365)

backtest-l2:
	PYTHONPATH=. uv run python scripts/backtest_l2.py --days $(if $(days),$(days),90)

backtest-l3:
	PYTHONPATH=. uv run python scripts/backtest_l3.py --symbol $(if $(symbol),$(symbol),NVDA) --days $(if $(days),$(days),30)

backtest-l4:
	PYTHONPATH=. uv run python scripts/backtest_l4.py --days $(if $(days),$(days),60)

# 整体回测命令
backtest:
	PYTHONPATH=. uv run python scripts/backtest.py $(if $(tf),$(tf),1h) --days $(if $(days),$(days),90) --top_n 1

# 环境与清理
setup:
	uv sync

clean:
	rm -rf models/artifacts/*.joblib
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf scripts/legacy/__pycache__
