.PHONY: train-1d train-1h train-15m backtest-1d backtest-1h backtest-15m predict help

# 默认目标
help:
	@echo "可用命令:"
	@echo "  make train-1d      - 训练日线排序模型 (Mag7 + Indices)"
	@echo "  make train-1h      - 训练小时线排序模型"
	@echo "  make train-15m     - 训练15分钟线排序模型"
	@echo "  make backtest-1d   - 运行日线策略回测 (默认Top 1)"
	@echo "  make backtest-1h   - 运行小时线策略回测"
	@echo "  make backtest-15m  - 运行15分钟线策略回测"
	@echo "  make predict-1d    - 运行当日预测 (日线)"
	@echo "  make setup         - 安装依赖"
	@echo "  make clean         - 清理输出文件"

# 训练命令
train-1d:
	uv run python main.py 1d

train-1h:
	uv run python main.py 1h

train-15m:
	uv run python main.py 15m

# 回测命令
backtest-1d:
	uv run python backtest.py 1d --days 365 --top_n 1

backtest-1h:
	uv run python backtest.py 1h --days 90 --top_n 1

backtest-15m:
	uv run python backtest.py 15m --days 30 --top_n 1

# 预测命令
predict-1d:
	uv run python predict.py 1d

# 环境与清理
setup:
	uv sync

clean:
	rm -rf output/*.joblib
	rm -rf __pycache__
	rm -rf */__pycache__
