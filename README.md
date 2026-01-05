# 算法交易系统 (Algo-Trade)

基于机器学习的四层架构量化交易系统,支持多空策略和智能风控。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│ L1: 市场择时 (Market Timing)                              │
│ • 模型: RandomForest Classifier                          │
│ • 输入: SPY, VIXY, TLT 宏观数据                           │
│ • 输出: 市场环境安全性评估                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ L2: 选股排序 (Stock Selection)                           │
│ • 模型: LGBMRanker                                       │
│ • 输入: 12 只科技股的技术指标 + SMC 特征                   │
│ • 输出: 排序得分 → 做多(Top) + 做空(Bottom)               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ L3: 执行信号 (Execution Signal)                          │
│ • 模型: LGBMClassifier (3分类)                           │
│ • 输入: 15分钟线技术指标 + 洗盘检测                        │
│ • 输出: Long/Short/Neutral 置信度                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ L4: 风控管理 (Risk Management)                           │
│ • 模型: 4 个 LGBMRegressor                               │
│ • 输入: 技术指标 + SMC 特征 (Swing High/Low, OB, FVG)    │
│ • 输出: 止盈止损点位 (做多/做空)                           │
└─────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 安装依赖

```bash
make setup
```

### 训练模型

```bash
# 训练所有模型
make train-l1  # 市场择时模型
make train-l2  # 选股排序模型
make train-l3  # 执行信号模型
make train-l4  # 风控管理模型
```

### 运行预测

```bash
make predict
```

### 🤖 全自动交易 (Paper Trading)

系统现已支持基于 Alpaca 模拟盘的全自动交易，包含自动止盈止损。

```bash
# 启动自动交易 (默认 15 分钟/间隔)
make trade

# 自定义检查间隔
make trade args="--interval 5"
```

**特性**:

- **支架订单 (Bracket Orders)**: 入场时自动根据 L4 模型预测挂好 TP/SL 订单。
- **模拟盘保护**: 强制在 Alpaca Paper Trading 环境运行。
- **分层过滤**: 仅在 L1 + L3 置信度达标时触发交易。

### 回测策略

```bash
# 默认: 1小时线, 90天, 多空策略, 显示交易明细
make backtest

# 自定义参数
make backtest tf=1h days=180
```

## 📊 核心特性

### 1. 多空策略

- **做多**: 选择排名最高的标的
- **做空**: 选择排名最低的标的
- **对冲收益**: `(做多收益 + 做空收益) / 2`

### 2. SMC (Smart Money Concepts) 特征

- **Swing High/Low**: 关键支撑阻力位
- **Order Blocks (OB)**: 机构订单区
- **Fair Value Gaps (FVG)**: 价格失衡区
- **Break of Structure (BOS)**: 市场结构突破
- **Displacement**: 大幅波动检测

### 3. 智能风控 (L4)

基于 SMC 概念预测止盈止损点位:

- 参考 Swing High/Low 作为支撑阻力
- 结合未来实际价格走势
- 自适应不同市场环境

### 4. 洗盘检测

- **Bullish Shakeout**: 假跌破后反弹
- **Bearish Trap**: 假突破后回落

## 📁 项目结构

```
algo-trade/
├── data/
│   └── provider.py          # Alpaca 数据接口
├── features/
│   ├── technical.py         # 技术指标 + SMC 特征
│   └── macro.py             # 宏观指标
├── models/
│   ├── trainer.py           # 模型训练器
│   └── artifacts/           # 训练好的模型
│       ├── l1_market_timing.joblib
│       ├── l2_stock_selection.joblib
│       ├── l3_execution.joblib
│       ├── l4_risk_tp_long.joblib
│       ├── l4_risk_sl_long.joblib
│       ├── l4_risk_tp_short.joblib
│       └── l4_risk_sl_short.joblib
├── scripts/
│   ├── train_l1.py          # L1 训练脚本
│   ├── train_l2.py          # L2 训练脚本
│   ├── train_l3.py          # L3 训练脚本
│   ├── train_l4.py          # L4 训练脚本
│   └── backtest.py          # 回测脚本
├── predict.py               # 实时预测
├── Makefile                 # 命令快捷方式
└── README.md
```

## 🔧 配置

在项目根目录创建 `.env` 文件:

```env
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
```

## 📈 回测示例

```bash
make backtest tf=1h days=90 --details
```

**输出示例**:

```
多空策略回测报告: 1h (Long 1 + Short 1) - ET
时间范围: 2024-10-05 至 2025-01-04
总周期数: 1500
--------------------------------------------------
策略累计收益 (Model): 15.32%
SPY 累计收益 (基准): 8.45%
QQQ 累计收益 (基准): 10.23%
最大回撤 (Max Drawdown): -5.67%
--------------------------------------------------
结论: 🏆 [策略成功跑赢所有基准!]
```

## 🎯 预测输出示例

```
[L2: 标的筛选分析]
📈 做多建议 (Top 3):
排名 | 代码   | 价格      | 相对强度得分
1    | NVDA   | 850.00    | 0.9234
2    | AMD    | 145.50    | 0.8567
3    | AVGO   | 920.30    | 0.8123

📉 做空建议 (Bottom 3):
排名 | 代码   | 价格      | 相对强度得分
1    | INTC   | 45.00     | -0.5123
2    | ORCL   | 125.00    | -0.3456
3    | MU     | 95.00     | -0.2789

[L4: 风控建议]
NVDA 做多:
  止损: $825.00 (-2.94%)
  止盈: $890.00 (+4.71%)
```

## 🛠️ 技术栈

- **数据源**: Alpaca Markets API
- **机器学习**: LightGBM, Scikit-learn
- **数据处理**: Pandas, NumPy
- **环境管理**: UV (Python package manager)

## 📝 开发路线图

- [x] L1 市场择时模型
- [x] L2 选股排序模型
- [x] L3 执行信号模型
- [x] L4 风控管理模型
- [x] 多空策略回测
- [x] SMC 特征工程
- [ ] 实盘交易接口
- [ ] 仓位管理优化
- [ ] Web 可视化界面

## ⚠️ 免责声明

本项目仅供学习和研究使用,不构成任何投资建议。实盘交易存在风险,请谨慎决策。

## 📄 许可证

MIT License
