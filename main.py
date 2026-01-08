import multiprocessing
import time
import uvicorn
from web.server import app
import trade
from data.streamer import MarketDataStreamer
from utils.logger import setup_logger

logger = setup_logger("main")

def start_trade_bot():
    """启动交易机器人进程"""
    logger.info("🚀 正在启动交易机器人 (Trade Bot)...")
    try:
        # 确保 trade.py 使用默认参数运行
        # 如果需要传参，可以修改 trade.main() 接受参数
        trade.main()
    except Exception as e:
        logger.error(f"❌ 交易机器人发生错误: {e}")

def start_web_server():
    """启动 Web Dashboard 进程"""
    logger.info("🌐 正在启动 Web Dashboard...")
    # 使用 uvicorn 启动 FastAPI 应用
    # host="0.0.0.0" 允许外部访问, port=8000
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def start_data_streamer():
    """启动实时数据流服务"""
    logger.info("📡 正在启动数据流服务 (Data Streamer)...")
    try:
        streamer = MarketDataStreamer()
        streamer.run()
    except Exception as e:
        logger.error(f"❌ 数据流服务发生错误: {e}")

def main():
    # 设置启动方式 (兼容 macOS/Windows)
    multiprocessing.set_start_method("spawn", force=True)
    
    # 创建子进程
    trade_process = multiprocessing.Process(target=start_trade_bot, name="TradeBot")
    web_process = multiprocessing.Process(target=start_web_server, name="WebServer")
    stream_process = multiprocessing.Process(target=start_data_streamer, name="DataStreamer")

    # 启动进程
    stream_process.start()
    time.sleep(2) # 等待 Streamer 先初始化
    trade_process.start()
    web_process.start()

    logger.info(f"✅ 服务已启动:")
    logger.info(f"   - Data Stream  PID: {stream_process.pid}")
    logger.info(f"   - Trade Bot    PID: {trade_process.pid}")
    logger.info(f"   - Web Server   PID: {web_process.pid}")
    logger.info(f"👉 Dashboard 地址: http://localhost:8000")

    try:
        # 主进程监控循环
        while True:
            time.sleep(1)
            
            # 检查进程是否存活
            if not stream_process.is_alive():
                 logger.warning("⚠️ 数据流服务进程意外退出!")
                 trade_process.terminate()
                 web_process.terminate()
                 break

            if not trade_process.is_alive():
                logger.warning("⚠️ 交易机器人进程意外退出!")
                web_process.terminate()
                stream_process.terminate()
                break
            
            if not web_process.is_alive():
                logger.warning("⚠️ Web Server 进程意外退出!")
                trade_process.terminate()
                stream_process.terminate()
                break
                
    except KeyboardInterrupt:
        logger.info("\n🛑 接收到停止指令, 正在停止所有服务...")
        trade_process.terminate()
        web_process.terminate()
        stream_process.terminate()
        
        trade_process.join()
        web_process.join()
        stream_process.join()
        logger.info("✅ 所有服务已安全停止。")

if __name__ == "__main__":
    main()
