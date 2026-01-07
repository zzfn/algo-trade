.PHONY: train-l1 train-l2 train-l3 train-l4 backtest predict trade help setup clean inspect-models

# 默认目标
help:
	@echo "============================================================"
	@echo "🚀 算法交易系统 - 可用命令"
	@echo "============================================================"
	@echo ""
	@echo "📊 模型训练:"
	@echo "  make train-l1          - 训练 L1 市场择时模型"
	@echo "  make train-l2          - 训练 L2 选股排序模型"
	@echo "  make train-l3          - 训练 L3 趋势确认模型"
	@echo "  make train-l4          - 训练 L4 收益预测模型"
	@echo "  make train-l5          - 训练 L5 元策略模型 ⭐"
	@echo ""
	@echo "🔧 超参数优化 (Optuna):"
	@echo "  make optimize-l2       - 优化 L2 模型参数"
	@echo "  make optimize-l3       - 优化 L3 模型参数"
	@echo "  make optimize-all      - 优化所有模型参数"
	@echo ""
	@echo "📈 回测:"
	@echo "  make backtest          - 运行传统回测"
	@echo "  make backtest-vbt      - VectorBT 回测 (30天) ⭐"
	@echo "  make backtest-vbt-full - VectorBT 完整回测 (90天)"
	@echo "  make backtest-l1       - L1 单层回测"
	@echo "  make backtest-l2       - L2 单层回测"
	@echo "  make backtest-l3       - L3 单层回测"
	@echo "  make backtest-l4       - L4 单层回测"
	@echo ""
	@echo "🤖 实时交易:"
	@echo "  make predict           - 运行实时预测"
	@echo "  make trade             - 启动自动交易"
	@echo ""
	@echo "🧠 L5 元策略:"
	@echo "  make generate-meta-data - 生成 L5 训练数据 (30-60分钟)"
	@echo "  make train-l5           - 训练 L5 元策略模型"
	@echo ""
	@echo "🛠️  其他:"
	@echo "  make inspect-models    - 分析模型特征重要性"
	@echo "  make setup             - 安装依赖"
	@echo "  make clean             - 清理输出文件"
	@echo ""
	@echo "============================================================"
	@echo "💡 提示: 使用 'make <命令> --help' 查看详细参数"
	@echo "============================================================"

# 训练命令
train-l1:
	PYTHONPATH=. uv run python scripts/train_l1.py

train-l2:
	PYTHONPATH=. uv run python scripts/train_l2.py

train-l3:
	PYTHONPATH=. uv run python scripts/train_l3.py

train-l4:
	PYTHONPATH=. uv run python scripts/train_l4.py

# 分析命令
inspect-models:
	PYTHONPATH=. uv run python scripts/inspect_features.py

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

# Optuna 超参数优化
optimize-l2:
	PYTHONPATH=. uv run python scripts/optimize_l2.py

optimize-l3:
	PYTHONPATH=. uv run python scripts/optimize_l3.py

optimize-all:
	make optimize-l2 && make optimize-l3

# VectorBT 回测
backtest-vbt:
	PYTHONPATH=. uv run python scripts/backtest_vbt.py --days $(if $(days),$(days),30)

backtest-vbt-full:
	PYTHONPATH=. uv run python scripts/backtest_vbt.py --days 90 --cash 100000

# L5 元策略
generate-meta-data:
	PYTHONPATH=. uv run python scripts/generate_meta_data.py --days 180

train-l5:
	PYTHONPATH=. uv run python scripts/train_l5.py

# Web Dashboard
.PHONY: dashboard

# 启动 Dashboard
dashboard:
	@echo "🚀 启动 Dashboard 服务器..."
	@echo "访问: http://localhost:8000"
	PYTHONPATH=. uv run uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload

# 启动完整系统 (交易机器人 + Dashboard)
run:
	PYTHONPATH=. uv run python main.py
