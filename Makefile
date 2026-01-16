# ============================================================
# 🚀 算法交易系统 Makefile
# ============================================================

.PHONY: help setup clean
.PHONY: train-l2 train-l3 train-return train-l5
.PHONY: optimize-l2 optimize-l3 optimize-all
.PHONY: backtest-l2 backtest-l3 backtest-l4
.PHONY: backtest-vbt backtest-vbt-full
.PHONY: run
.PHONY: generate-meta-data inspect-models

# ============================================================
# 📖 帮助信息
# ============================================================

help:
	@echo "============================================================"
	@echo "🚀 算法交易系统 - 可用命令"
	@echo "============================================================"
	@echo ""
	@echo "🔧 环境管理:"
	@echo "  make setup             - 安装项目依赖"
	@echo "  make clean             - 清理缓存和临时文件"
	@echo ""
	@echo "📊 模型训练:"
	@echo "  make train-l2          - 训练 L2 选股排序模型"

	@echo "  make train-l3          - 训练 L3 趋势确认模型"
	@echo "  make train-return      - 训练收益预测模型"
	@echo "  make train-l5          - 训练 L5 元策略模型 ⭐"
	@echo ""
	@echo "🔍 超参数优化:"
	@echo "  make optimize-l2       - 优化 L2 模型参数 (Optuna)"
	@echo "  make optimize-l3       - 优化 L3 模型参数 (Optuna)"
	@echo "  make optimize-all      - 优化所有模型参数"
	@echo ""
	@echo "📈 策略回测:"
	@echo "  make backtest-vbt      - VectorBT 快速回测 (30天) ⭐"
	@echo "  make backtest-vbt-full - VectorBT 完整回测 (90天)"
	@echo "  make backtest-l2       - L2 单层回测 (90天)"
	@echo "  make backtest-l3       - L3 单层回测 (30天)"
	@echo "  make backtest-l4       - L4 单层回测 (60天)"
	@echo ""
	@echo "🤖 实时交易:"
	@echo "  make run               - 启动完整系统"
	@echo ""
	@echo "🛠️  工具命令:"
	@echo "  make inspect-models    - 分析模型特征重要性"
	@echo "  make generate-meta-data - 生成 L5 训练数据 (30-60分钟)"
	@echo ""
	@echo "============================================================"
	@echo "💡 提示: 部分命令支持参数,如 'make backtest days=30'"
	@echo "============================================================"

# ============================================================
# 🔧 环境管理
# ============================================================

setup:
	@echo "📦 安装项目依赖..."
	uv sync

clean:
	@echo "🧹 清理缓存文件..."
	rm -rf models/artifacts/*.joblib
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf scripts/legacy/__pycache__

# ============================================================
# 📊 模型训练
# ============================================================

train-l2:
	PYTHONPATH=. uv run python scripts/train_l2.py

train-l3:
	PYTHONPATH=. uv run python scripts/train_l3.py

train-return:
	PYTHONPATH=. uv run python scripts/train_l4.py

train-l5:
	PYTHONPATH=. uv run python scripts/train_l5.py

# ============================================================
# 🔍 超参数优化 (Optuna)
# ============================================================

optimize-l2:
	PYTHONPATH=. uv run python scripts/optimize_l2.py

optimize-l3:
	PYTHONPATH=. uv run python scripts/optimize_l3.py

optimize-all:
	@echo "🔍 开始优化所有模型..."
	make optimize-l2 && make optimize-l3

# ============================================================
# 📈 策略回测
# ============================================================

# VectorBT 回测 (推荐 - 快速)
backtest-vbt:
	PYTHONPATH=. uv run python scripts/backtest_vbt.py --days $(if $(days),$(days),1)

backtest-vbt-full:
	PYTHONPATH=. uv run python scripts/backtest_vbt.py --days 90 --cash 100000

# 分层回测
backtest-l2:
	PYTHONPATH=. uv run python scripts/backtest_l2.py --days $(if $(days),$(days),90)

backtest-l3:
	PYTHONPATH=. uv run python scripts/backtest_l3.py --symbol $(if $(symbol),$(symbol),NVDA) --days $(if $(days),$(days),30)

backtest-l4:
	PYTHONPATH=. uv run python scripts/backtest_l4.py --days $(if $(days),$(days),60)

# ============================================================
# 🤖 实时交易
# ============================================================

run:
	@echo "🚀 启动完整交易系统..."
	PYTHONPATH=. uv run python main.py

# ============================================================
# 🛠️  工具命令
# ============================================================

inspect-models:
	PYTHONPATH=. uv run python scripts/inspect_features.py

generate-meta-data:
	@echo "🧠 生成 L5 元策略训练数据 (预计 30-60 分钟)..."
	PYTHONPATH=. uv run python scripts/generate_meta_data.py --days 180
