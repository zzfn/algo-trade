import os
import asyncio
import signal
import pandas as pd
from datetime import datetime
from loguru import logger
from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv

from models.constants import L2_SYMBOLS
from data.redis_manager import RedisDataManager

# 配置日志
logger.add("logs/streamer.log", rotation="100 MB", level="INFO")

class MarketDataStreamer:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Keys (ALPACA_API_KEY, ALPACA_SECRET_KEY) must be set in environment.")

        # 初始化 Redis Manager
        self.redis_mgr = RedisDataManager()
        
        # 初始化 Alpaca Data Stream (使用 IEX/SIP 需根据订阅情况)
        # 注意: paper=True 在 Live Client 中不适用，Data Stream 根据 API Key 权限决定数据源
        # 显式指定 feed='iex' 用于免费数据或测试，生产环境如有权限可用 'sip'
        self.stream_client = StockDataStream(self.api_key, self.secret_key, feed=DataFeed.IEX)
        
        # 订阅列表
        self.symbols = L2_SYMBOLS
        logger.info(f"📋 订阅标的 ({len(self.symbols)}): {self.symbols}")

    async def bar_handler(self, bar):
        """处理接收到的 1分钟 K线数据"""
        try:
            # bar 是 alpaca.data.models.Bar 对象
            symbol = bar.symbol
            timestamp = bar.timestamp # UTC Aware datetime
            
            logger.info(f"📡 接收到 Bar: {symbol} @ {timestamp.strftime('%H:%M')} | Close: {bar.close}")
            
            # 转换为 DataFrame 格式以适配 RedisDataManager
            # 注意: RedisDataManager 会处理时区问题, 这里传入原始 UTC 时间戳即可
            data = {
                'timestamp': timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'trade_count': bar.trade_count,
                'vwap': bar.vwap,
                'symbol': symbol
            }
            
            df = pd.DataFrame([data])
            
            # 写入 Redis
            # 这里的 timeframe 默认为 1Minute，因为 StockDataStream 发送的是分钟线
            from alpaca.data.timeframe import TimeFrame
            self.redis_mgr.save_bars(df, symbol, TimeFrame.Minute)
            
        except Exception as e:
            logger.error(f"❌ 处理 Bar 数据失败: {e}", exc_info=True)

    def run(self):
        """启动 Stream Loop"""
        logger.info("🚀 启动 Market Data Streamer (Websocket)...")
        
        # 注册 Bar Handler
        # 使用 subscribe_bars 订阅分钟线
        self.stream_client.subscribe_bars(self.bar_handler, *self.symbols)
        
        try:
            # 运行 Event Loop (SDK 内部封装了 run 方法)
            self.stream_client.run()
        except KeyboardInterrupt:
            logger.info("🛑 Streamer 接收到停止信号")
        except Exception as e:
            logger.error(f"❌ Streamer 异常退出: {e}", exc_info=True)
            # 在某些网络错误下，这里可能需要重试逻辑，Alpaca SDK 有自动重连，但如果彻底断开需要外部重启
            # 简单的进程重启交由 main.py 的 process 监控处理

if __name__ == "__main__":
    streamer = MarketDataStreamer()
    streamer.run()
