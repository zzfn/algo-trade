#!/usr/bin/env python3
"""测试预测引擎初始化"""

import sys
import traceback

print("=" * 60)
print("测试预测引擎初始化")
print("=" * 60)

try:
    print("\n1. 导入 StrategyEngine...")
    from models.engine import StrategyEngine
    print("   ✅ 导入成功")
    
    print("\n2. 初始化 StrategyEngine...")
    engine = StrategyEngine()
    print("   ✅ 初始化成功")
    
    print("\n3. 测试预测功能...")
    from datetime import datetime
    import pytz
    
    ny_tz = pytz.timezone("America/New_York")
    target_dt = datetime.now(ny_tz).replace(tzinfo=None)
    
    print(f"   使用时间: {target_dt}")
    print("   正在运行 analyze()...")
    
    results = engine.analyze(target_dt)
    
    print(f"   ✅ 分析完成")
    print(f"   - L1 Safe: {results.get('l1_safe')}")
    print(f"   - L1 Prob: {results.get('l1_prob'):.2%}")
    print(f"   - L2 Timestamp: {results.get('l2_timestamp')}")
    print(f"   - L3 Timestamp: {results.get('l3_timestamp')}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！预测引擎工作正常。")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    print("\n详细错误信息:")
    print(traceback.format_exc())
    print("\n" + "=" * 60)
    print("❌ 测试失败")
    print("=" * 60)
    sys.exit(1)
