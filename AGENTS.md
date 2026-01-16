# AGENTS.md - 算法交易系统开发指南

本文档为 AI 编程助手提供在此算法交易系统中工作的具体指南。

## 🚀 构建/测试/代码质量命令

### 开发环境管理

```bash
# 安装项目依赖
make setup
# 或
uv sync

# 清理缓存
make clean
```

### 运行测试

```bash

# 运行预测功能测试
PYTHONPATH=. uv run python predict.py --help

```

### 代码质量检查

```bash
# 目前项目使用 Python 类型提示，建议运行以下检查：
PYTHONPATH=. uv run python -m py_compile trade.py  # 语法检查
```

### 模型训练与优化

```bash
# 训练各层模型
make train-l1    # L1 市场择时
make train-l2    # L2 选股排序
make train-l3    # L3 趋势确认
make train-l4    # L4 收益预测
make train-l5    # L5 元策略

# 超参数优化
make optimize-l2
make optimize-l3
```

### 回测验证

```bash
# 快速回测 (推荐)
make backtest-vbt days=30

# 完整回测
make backtest-vbt-full

# 分层回测
make backtest-l1 days=365
make backtest-l2 days=90
make backtest-l3 symbol=NVDA days=30
```

### 实时系统

```bash
# 运行预测引擎
make predict

# 启动交易机器人 (模拟盘)
make trade

# 启动完整系统 (机器人 + Dashboard)
make run

# 仅启动 Dashboard
make dashboard
```

## 📝 代码风格指南

### 导入顺序

```python
# 标准库导入
import os
import time
from datetime import datetime

# 第三方库导入
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient

# 本地模块导入
from models.engine import StrategyEngine
from utils.logger import setup_logger
```

### 类型提示

所有函数都应该包含类型提示：

```python
def calculate_position_size(target_value: float, price: float) -> int:
    """计算仓位大小"""
    return int(target_value / price)
```

### 命名约定

- **类名**: `PascalCase` (如 `TradingBot`, `StrategyEngine`)
- **函数/变量**: `snake_case` (如 `run_iteration`, `target_dt`)
- **常量**: `UPPER_SNAKE_CASE` (如 `SIGNAL_THRESHOLD`, `TOP_N_TRADES`)
- **私有方法**: 以下划线开头 (如 `_validate_account_data`)

### 错误处理

```python
# 推荐的具体异常处理
try:
    account = self.trading_client.get_account()
except ConnectionError as e:
    logger.error(f"API连接失败: {e}")
    return None
except Exception as e:
    logger.error(f"获取账户信息失败: {e}")
    raise

# 避免:
try:
    # 业务逻辑
except Exception:
    pass  # 不要掩盖所有错误
```

### 日志记录

使用统一的日志配置：

```python
from utils.logger import setup_logger

logger = setup_logger("module_name")

logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
logger.debug("调试日志")
```

### 金融数据处理

```python
# 使用 Decimal 处理金融计算以提高精度
from decimal import Decimal

price = Decimal("150.25")
quantity = Decimal("10")
cost = price * quantity

# 避免 float 在关键计算中
# 不推荐: cost = 150.25 * 10.0
```

## 🏗️ 项目架构

### 目录结构

```
algo-trade/
├── data/           # 数据层 - 数据获取、存储、预处理
├── features/       # 特征工程 - 技术指标、宏观特征
├── models/         # 模型层 - 策略引擎、模型管理
├── scripts/        # 脚本 - 训练、回测、优化
├── utils/          # 工具 - 日志、辅助函数
├── web/           # Web界面 - Dashboard、API服务
├── trade.py       # 主交易机器人
├── predict.py     # 预测引擎
└── main.py        # 系统入口
```

### 配置管理

- 所有常量定义在 `models/constants.py`
- 动态参数通过 `models/dynamic_params.py` 管理
- 环境变量使用 `python-dotenv` 加载

### 数据规范

- 时间统一使用 ET 纽约时间
- 数据通过 `DataProvider` 统一获取
- 特征列通过 `get_feature_columns()` 过滤

## 🔒 安全注意事项

### API 密钥

```python
# 必须验证密钥存在
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
if not api_key or not secret_key:
    raise ValueError("API密钥未配置")
```

### 交易安全

- 生产环境始终使用 `paper=True` 模拟盘
- 所有交易操作必须记录日志
- 关键操作需要多重验证

## 🧪 测试策略

### 单元测试

为关键函数编写测试：

- 风险参数计算
- 仓位分配逻辑
- 信号生成逻辑

### 集成测试

- 完整交易流程测试
- 数据管道验证
- 模型推理测试

### 回测验证

- 使用历史数据验证策略
- 多时间段 robustness 检查
- 极端市场条件测试

## ⚡ 性能优化

### API 调用优化

- 缓存账户和持仓信息
- 批量获取市场数据
- 使用 Redis 缓存重复查询

### 数据处理优化

- 使用向量化操作替代循环
- 避免重复的特征计算
- 合理使用数据类型

## 📋 开发检查清单

提交代码前确认：

- [ ] 所有函数有类型提示
- [ ] 错误处理完善
- [ ] 日志记录规范
- [ ] 金融计算使用 Decimal
- [ ] 通过基本语法检查
- [ ] 敏感信息使用环境变量
- [ ] 遵循导入顺序规范
- [ ] 方法长度合理 (<50 行)

## 🚨 重要提醒

1. **永远不要提交 API 密钥到版本控制**
2. **生产代码必须经过充分回测验证**
3. **修改交易逻辑前需要在沙盒环境测试**
4. **保持日志记录完整性，便于调试和审计**
5. **遵循现有架构模式，不要过度设计**
