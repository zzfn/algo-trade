import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Union
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

class DataProvider:
    @staticmethod
    def get_tf_string(tf: TimeFrame) -> str:
        """
        将 Alpaca TimeFrame 转换为业界通用字符串 (如 1d, 15m, 1h)
        """
        unit_map = {
            TimeFrameUnit.Minute: 'm',
            TimeFrameUnit.Hour: 'h',
            TimeFrameUnit.Day: 'd',
            TimeFrameUnit.Week: 'w',
            TimeFrameUnit.Month: 'M'
        }
        unit_str = unit_map.get(tf.unit, 'u')
        return f"{tf.amount}{unit_str}"

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Key and Secret Key must be provided or set as environment variables.")
        
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # 缓存配置
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_bars(self, symbols: Union[str, List[str]], timeframe: TimeFrame, start: datetime, end: Optional[datetime] = None, use_cache: bool = True, use_redis: bool = False) -> pd.DataFrame:
        """
        Fetch historical bars with local caching (File or Redis).
        
        Args:
            use_redis: 如果为 True,则尝试使用 Redis 进行增量更新
        """
        if end is None:
            # 强制将本地时间视为北京时间 (Asia/Shanghai)
            # 这是为了解决用户系统时区设置不正确的问题
            import pytz
            local_naive = datetime.now()
            beijing_tz = pytz.timezone('Asia/Shanghai')
            # 假定本地时间就是北京时间
            local_aware = beijing_tz.localize(local_naive)
            # 转为 UTC 供后续使用
            end_aware = local_aware.astimezone(pytz.utc)
            
            # 检查 start 是否为 Naive (通常意味着是 NY Time)
            if start.tzinfo is None:
                # 将 end 转为 Naive NY Time 以匹配 start
                ny_tz = pytz.timezone('America/New_York')
                end = end_aware.astimezone(ny_tz).replace(tzinfo=None)
            else:
                end = end_aware

        # --- Redis 增量更新逻辑 ---
        if use_redis:
            try:
                from data.redis_manager import RedisDataManager
                redis_mgr = RedisDataManager()
                
                # 统一处理单个和多个标的
                sym_list = [symbols] if isinstance(symbols, str) else symbols
                
                # 1. 批量检查所有标的的最新时间
                # 为了批量调用的效率,我们找到"最旧"的最新时间,一次性拉取所有增量
                # (虽然这可能会重复拉取一些数据,但比发 N 个 API 请求要快得多且省额度)
                active_start_time = start
                
                # 检查每个标的在 Redis 的最新时间
                latest_timestamps = []
                for sym in sym_list:
                    ts = redis_mgr.get_latest_timestamp(sym, timeframe)
                    if ts:
                        latest_timestamps.append(ts)
                    else:
                        latest_timestamps.append(None)
                
                # 如果所有标的都有缓存数据,找到最早的一个作为增量起点
                if all(ts is not None for ts in latest_timestamps):
                    min_ts = min(latest_timestamps)
                    
                    # 转换 min_ts 为 ET 用于显示
                    import pytz
                    ny_tz = pytz.timezone('America/New_York')
                    show_min_ts = min_ts.astimezone(ny_tz) if min_ts.tzinfo else pytz.utc.localize(min_ts).astimezone(ny_tz)
                    show_start_chk = start.astimezone(ny_tz) if start.tzinfo else pytz.utc.localize(start).astimezone(ny_tz)

                    if min_ts >= start:
                         # 增量起点: 最早的缓存时间 + 1分钟 (防止重叠)
                        active_start_time = min_ts + timedelta(minutes=1)
                        print(f"🔄 Redis 批量增量: 本地数据均新于 {show_min_ts.strftime('%Y-%m-%d %H:%M:%S')} ET, 仅拉取增量...")
                    else:
                        print(f"📥 Redis 数据较旧 (部分旧于 {show_start_chk.strftime('%Y-%m-%d %H:%M:%S')} ET), 拉取完整历史...")
                else:
                    print(f"📥 Redis 部分标的缺数据, 拉取完整历史...")

                # 2. 从 API 批量拉取数据 (如果需要)
                if active_start_time < end:
                    # 转换显示时间为 ET
                    import pytz
                    ny_tz = pytz.timezone('America/New_York')
                    
                    show_start = active_start_time
                    if show_start.tzinfo is None:
                        # Assumed to be Naive NY Time (based on project convention)
                        import pytz
                        ny_tz = pytz.timezone('America/New_York')
                        show_start = ny_tz.localize(show_start)
                    else:
                        show_start = show_start.astimezone(ny_tz)
                        
                    show_end = end
                    if show_end.tzinfo is None:
                        show_end = pytz.utc.localize(show_end).astimezone(ny_tz)
                    else:
                        show_end = show_end.astimezone(ny_tz)
                        
                    print(f"DEBUG TIME: Now(UTC)={end} | End(ET)={show_end}")
                    print(f"⬇️  Fetching batch data from API ({show_start.strftime('%Y-%m-%d %H:%M:%S')} ET -> {show_end.strftime('%Y-%m-%d %H:%M:%S')} ET)...")
                    request_params = StockBarsRequest(
                        symbol_or_symbols=sym_list,
                        timeframe=timeframe,
                        start=active_start_time,
                        end=end,
                        feed=DataFeed.IEX
                    )
                    try:
                        bars = self.client.get_stock_bars(request_params)
                        new_df = bars.df
                        print(f"✅ API returned {len(new_df)} rows of data.")
                        
                        if not new_df.empty:
                            print(f"🔍 API Data Preview:\n{new_df.iloc[[0, -1]][['timestamp']] if 'timestamp' in new_df.columns else new_df.index[[0, -1]]}")

                            # 统一格式处理
                            if isinstance(new_df.index, pd.MultiIndex):
                                new_df = new_df.reset_index()
                            else:
                                new_df = new_df.reset_index()
                                # 如果API返回单标的格式但我们请求的是列表(极端情况),补全 symbol
                                if 'symbol' not in new_df.columns and len(sym_list) == 1:
                                     new_df['symbol'] = sym_list[0]

                            if 'timestamp' in new_df.columns:
                                new_df['timestamp'] = pd.to_datetime(new_df['timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
                            
                            # 3. 按标的分组并保存到 Redis
                            grouped = new_df.groupby('symbol')
                            for sym, group in grouped:
                                redis_mgr.save_bars(group, sym, timeframe)
                                
                    except Exception as e:
                        import traceback
                        print(f"⚠️  Batch fetch failed (maybe no new data): {e}")
                        print(traceback.format_exc())

                # 4. 从 Redis 组装完整数据集返回
                import pytz
                ny_tz = pytz.timezone('America/New_York')
                
                show_start_full = start
                if show_start_full.tzinfo is None:
                     show_start_full = pytz.utc.localize(show_start_full).astimezone(ny_tz)
                else:
                     show_start_full = show_start_full.astimezone(ny_tz)

                show_end_full = end
                if show_end_full.tzinfo is None:
                     show_end_full = pytz.utc.localize(show_end_full).astimezone(ny_tz)
                else:
                     show_end_full = show_end_full.astimezone(ny_tz)

                print(f"📦 Loading full batch dataset from Redis ({show_start_full.strftime('%Y-%m-%d %H:%M:%S')} ET -> {show_end_full.strftime('%Y-%m-%d %H:%M:%S')} ET)...")
                all_data = []
                for sym in sym_list:
                    df_sym = redis_mgr.get_bars(sym, timeframe, start, end)
                    if not df_sym.empty:
                        df_sym['symbol'] = sym
                        all_data.append(df_sym)
                
                if all_data:
                    full_df = pd.concat(all_data, ignore_index=True)
                    # 确保格式符合预期 (symbol, timestamp) MultiIndex 或 Column
                    # 这里返回 flat DataFrame, 让调用者处理
                    return full_df
                else:
                    return pd.DataFrame()
                
            except ImportError:
                print("⚠️  Redis dependencies not installed. Falling back to file cache.")
            except Exception as e:
                print(f"⚠️  Redis batch operation failed: {e}. Falling back to file cache.")

        # --- 原有的文件缓存逻辑 (Fallback) ---
        # 1. 生成缓存文件名
        # 格式: timeframe_start_end_hash.parquet
        sym_str = symbols if isinstance(symbols, str) else "_".join(sorted(symbols))
        # 如果 symbol 太多，使用 hash 避免文件名过长
        if len(sym_str) > 50:
            import hashlib
            sym_str = hashlib.md5(sym_str.encode()).hexdigest()
            
        tf_str = self.get_tf_string(timeframe)
        start_str = start.strftime('%Y%m%d')
        end_str = end.strftime('%Y%m%d')
        
        cache_file = os.path.join(self.cache_dir, f"{sym_str}_{tf_str}_{start_str}_{end_str}.parquet")
        
        # 2. 尝试读取缓存
        if use_cache and os.path.exists(cache_file):
            try:
                print(f"📦 Loading cached data from {cache_file}...")
                df = pd.read_parquet(cache_file)
                return df
            except Exception as e:
                print(f"⚠️  Cache load failed, fetching from API: {e}")

        # 3. 从 API 获取数据
        print(f"⬇️  Fetching data from Alpaca API ([{sym_str}] {tf_str})...")
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX
        )
        
        bars = self.client.get_stock_bars(request_params)
        df = bars.df
        
        # 处理多索引
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        else:
            # 如果是单标的，手动加上 symbol 列保持格式统一
            df = df.reset_index()
            if isinstance(symbols, str):
                df['symbol'] = symbols
        
        # 确保 timestamp 列转换为美东时间 (America/New_York)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None)
            
        # 4. 保存缓存
        if use_cache and not df.empty:
            try:
                df.to_parquet(cache_file, index=False)
                print(f"💾 Data cached to {cache_file}")
            except Exception as e:
                print(f"⚠️  Cache save failed: {e}")
            
        return df

if __name__ == "__main__":
    # Example usage (will fail if keys not set)
    try:
        provider = DataProvider()
        # Fetch last 30 days of daily data for QQQ
        end = datetime.now()
        start = end - timedelta(days=30)
        data = provider.fetch_bars("QQQ", TimeFrame.Day, start, end)
        print(data.head())
    except Exception as e:
        print(f"Error fetching data: {e}")
