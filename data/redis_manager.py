
import os
import json
import pandas as pd
import redis
import numpy as np
from datetime import datetime
from typing import Optional, List, Union
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

class RedisDataManager:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        # 优先使用环境变量配置
        host = os.getenv("REDIS_HOST", host)
        port = int(os.getenv("REDIS_PORT", port))
        password = os.getenv("REDIS_PASSWORD", password)
        
        self.redis = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            password=password, 
            decode_responses=True
        )
        
    def get_key(self, symbol: str, timeframe: TimeFrame) -> str:
        """生成 Redis Key: market_data:{symbol}:{tf}"""
        tf_str = f"{timeframe.amount}{timeframe.unit.value}"
        return f"market_data:{symbol}:{tf_str}"

    def get_latest_timestamp(self, symbol: str, timeframe: TimeFrame) -> Optional[datetime]:
        """获取 Redis 中存储的最晚时间戳"""
        key = self.get_key(symbol, timeframe)
        # 获取 ZSET 中最后一个元素 (分数为时间戳)
        result = self.redis.zrange(key, -1, -1, withscores=True)
        if result:
            _, score = result[0]
            # score 是 UTC 时间戳
            import pytz
            utc_dt = datetime.fromtimestamp(score, tz=pytz.utc)
            ny_tz = pytz.timezone('America/New_York')
            ny_dt = utc_dt.astimezone(ny_tz)
            return ny_dt.replace(tzinfo=None) # 返回 Naive NY Time
        return None

    def save_bars(self, df: pd.DataFrame, symbol: str, timeframe: TimeFrame):
        """保存 K 线数据到 Redis ZSET"""
        if df.empty:
            return
            
        key = self.get_key(symbol, timeframe)
        pipeline = self.redis.pipeline()
        
        # 确保 timestamp 列存在
        if 'timestamp' in df.columns:
             # 确保是 datetime 类型
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # 假设 index 是 timestamp
            df = df.reset_index()
            df.rename(columns={'index': 'timestamp'}, inplace=True)

        for _, row in df.iterrows():
            ts = row['timestamp']
            
            # 统一转为 UTC 时间戳作为 score
            # 假设 ts 是 Naive NY Time (项目约定), 先 localize 到 NY, 再转 UTC
            if ts.tzinfo is None:
                # 只有当它是 naive 时才当作 NY Time 处理
                import pytz
                ny_tz = pytz.timezone('America/New_York')
                ts_aware = ny_tz.localize(ts)
                timestamp_score = ts_aware.timestamp() # .timestamp() returns UTC timestamp for aware obj
            else:
                timestamp_score = ts.timestamp()
            
            # 序列化数据 (去除 timestamp 字段,因为它是 score)
            data_dict = row.drop('timestamp').to_dict()
            # 处理 timestamp 对象无法 JSON 序列化的问题
            data_json = json.dumps(data_dict, default=str)
            
            # ZADD key score member
            pipeline.zadd(key, {data_json: timestamp_score})
            
        pipeline.execute()
        print(f"💾 Saved {len(df)} bars to Redis: {key}")

    def get_bars(self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime) -> pd.DataFrame:
        """从 Redis 获取指定时间范围的数据"""
        key = self.get_key(symbol, timeframe)
        
        start_score = start.timestamp()
        end_score = end.timestamp()
        
        # ZRANGEBYSCORE key min max
        results = self.redis.zrangebyscore(key, start_score, end_score, withscores=True)
        
        if not results:
            return pd.DataFrame()
            
        data_list = []
        for member, score in results:
            data = json.loads(member)
            # score 是 UTC 时间戳
            import pytz
            utc_dt = datetime.fromtimestamp(score, tz=pytz.utc)
            ny_tz = pytz.timezone('America/New_York')
            ny_dt = utc_dt.astimezone(ny_tz)
            data['timestamp'] = ny_dt.replace(tzinfo=None) # 返回 Naive NY Time
            data_list.append(data)
            
        df = pd.DataFrame(data_list)
        return df
