"""
Web Dashboard 状态管理器

负责从 Redis 或文件读取交易状态数据
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import redis
from loguru import logger


class StateManager:
    """管理交易状态的读取和缓存"""
    
    def __init__(self):
        self.redis_client = None
        self.fallback_file = Path("data/trading_state.json")
        self.cache: Optional[Dict[str, Any]] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl = 5  # 缓存 5 秒
        
        # 尝试连接 Redis
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            logger.info(f"✅ Redis 连接成功: {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"⚠️ Redis 连接失败: {e}, 将使用文件备份")
            self.redis_client = None
    
    def get_state(self) -> Optional[Dict[str, Any]]:
        """获取最新交易状态"""
        # 检查缓存
        if self.cache and self.cache_timestamp:
            age = (datetime.now() - self.cache_timestamp).total_seconds()
            if age < self.cache_ttl:
                return self.cache
        
        # 尝试从 Redis 读取
        if self.redis_client:
            try:
                data = self.redis_client.get("trading:state")
                if data:
                    state = json.loads(data)
                    self.cache = state
                    self.cache_timestamp = datetime.now()
                    return state
            except Exception as e:
                logger.warning(f"⚠️ Redis 读取失败: {e}")
        
        # Fallback 到文件
        return self._read_from_file()
    
    def _read_from_file(self) -> Optional[Dict[str, Any]]:
        """从文件读取状态"""
        try:
            if self.fallback_file.exists():
                with open(self.fallback_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.cache = state
                    self.cache_timestamp = datetime.now()
                    return state
        except Exception as e:
            logger.error(f"❌ 文件读取失败: {e}")
        
        return None
    
    def get_account(self) -> Dict[str, Any]:
        """获取账户信息"""
        state = self.get_state()
        if state and "account" in state:
            return state["account"]
        return {
            "equity": 0.0,
            "cash": 0.0,
            "buying_power": 0.0,
            "portfolio_value": 0.0
        }
    
    def get_positions(self) -> list:
        """获取持仓列表"""
        state = self.get_state()
        if state and "positions" in state:
            return state["positions"]
        return []
    
    def get_signals(self) -> Dict[str, list]:
        """获取交易信号"""
        state = self.get_state()
        if state and "signals" in state:
            return state["signals"]
        return {"long": [], "short": []}
    
    def get_orders(self) -> list:
        """获取订单历史"""
        state = self.get_state()
        if state and "orders" in state:
            return state["orders"]
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        state = self.get_state()
        if state and "status" in state:
            return state["status"]
        return {
            "last_update": None,
            "is_market_open": False,
            "l1_safe": False,
            "l1_prob": 0.0,
            "is_running": False
        }
