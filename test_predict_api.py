#!/usr/bin/env python3
"""直接测试预测API的逻辑"""

import sys
import traceback
from datetime import datetime
import pytz

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("测试预测API逻辑")
print("=" * 60)

try:
    print("\n1. 初始化 StrategyEngine...")
    from models.engine import StrategyEngine
    engine = StrategyEngine()
    print("   ✅ 成功")
    
    print("\n2. 准备预测参数...")
    ny_tz = pytz.timezone("America/New_York")
    target_dt = datetime.now(ny_tz).replace(tzinfo=None)
    print(f"   时间: {target_dt}")
    
    print("\n3. 执行 analyze()...")
    results = engine.analyze(target_dt)
    print("   ✅ 成功")
    
    print("\n4. 构建响应数据...")
    response = {
        "timestamp": target_dt.isoformat(),
        "l1_safe": results.get('l1_safe', False),
        "l1_prob": float(results.get('l1_prob', 0)),
        "l2_timestamp": results.get('l2_timestamp').isoformat() if results.get('l2_timestamp') else None,
        "l3_timestamp": results.get('l3_timestamp').isoformat() if results.get('l3_timestamp') else None,
    }
    
    print("   ✅ 成功")
    print(f"\n响应数据:")
    import json
    print(json.dumps(response, indent=2))
    
    print("\n" + "=" * 60)
    print("✅ 预测API逻辑测试通过")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    print("\n详细错误:")
    print(traceback.format_exc())
    sys.exit(1)
