"""
测试市场开放检查功能

运行方式:
    PYTHONPATH=. uv run python scripts/test_market_check.py
"""

import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

def test_market_status():
    """测试市场状态检查"""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    trading_client = TradingClient(api_key, secret_key, paper=True)
    ny_tz = pytz.timezone("America/New_York")
    
    print("=" * 60)
    print("🧪 测试市场开放检查功能")
    print("=" * 60)
    
    try:
        # 获取市场时钟
        clock = trading_client.get_clock()
        
        # 当前时间
        now_et = datetime.now(ny_tz)
        print(f"\n📅 当前时间 (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # 市场状态
        print(f"\n📊 市场状态:")
        print(f"   是否开放: {'✅ 是' if clock.is_open else '❌ 否'}")
        
        # 下次开盘时间
        if clock.next_open:
            next_open_et = clock.next_open.astimezone(ny_tz)
            print(f"   下次开盘: {next_open_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # 计算等待时间
            wait_seconds = (next_open_et - now_et).total_seconds()
            wait_hours = wait_seconds / 3600
            print(f"   距离开盘: {wait_hours:.1f} 小时")
        
        # 下次收盘时间
        if clock.next_close:
            next_close_et = clock.next_close.astimezone(ny_tz)
            print(f"   下次收盘: {next_close_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # 当前交易日
        print(f"\n📆 交易日信息:")
        print(f"   时间戳: {clock.timestamp}")
        
        print("\n" + "=" * 60)
        print("✅ 测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_market_status()
