"""
统一日志配置模块

提供标准化的日志配置，支持控制台和文件输出。
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "algo_trade", level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    配置并返回日志记录器。
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 可选的日志文件路径
        
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复配置
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 强制日志时间使用美东时间
    import pytz
    ny_tz = pytz.timezone('America/New_York')
    
    # 必须将其包装为静态方法，否则赋值给类属性后，实列调用时会自动传入 self (Formatter对象)
    # 导致 et_converter(self, seconds) -> fromtimestamp(self) -> TypeError
    def et_converter(seconds):
        return datetime.fromtimestamp(seconds, ny_tz).timetuple()
        
    logging.Formatter.converter = staticmethod(et_converter)

    # 控制台输出格式 (带颜色和 emoji 友好)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="%H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 可选: 文件输出 (更详细的格式)
    if log_file:
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "algo_trade") -> logging.Logger:
    """
    获取已配置的日志记录器。
    如果尚未配置，则使用默认设置初始化。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
