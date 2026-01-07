"""
Web Dashboard FastAPI 服务器

提供 RESTful API 和静态文件服务
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from web.state_manager import StateManager
from models.engine import StrategyEngine
from models.constants import TOP_N_TRADES, SIGNAL_THRESHOLD

app = FastAPI(
    title="Algo Trade Dashboard",
    description="量化交易实时监控 Dashboard",
    version="1.0.0"
)

# CORS 配置 (开发阶段)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化状态管理器
state_manager = StateManager()

# 初始化预测引擎
strategy_engine = None
strategy_engine_error = None
try:
    print("正在初始化预测引擎...")
    strategy_engine = StrategyEngine()
    print("✅ 预测引擎初始化成功")
except Exception as e:
    import traceback
    error_msg = f"⚠️ 预测引擎初始化失败: {e}"
    print(error_msg)
    print(traceback.format_exc())
    strategy_engine_error = str(e)
    strategy_engine = None


# 静态文件目录
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ==================== API 端点 ====================

@app.get("/api/account")
async def get_account():
    """获取账户信息"""
    try:
        return state_manager.get_account()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions")
async def get_positions():
    """获取当前持仓"""
    try:
        return state_manager.get_positions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals")
async def get_signals():
    """获取交易信号"""
    try:
        return state_manager.get_signals()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders")
async def get_orders():
    """获取订单历史"""
    try:
        return state_manager.get_orders()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """获取系统运行状态"""
    try:
        return state_manager.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 预测与模型调试 API ====================

@app.get("/api/predict")
async def run_prediction():
    """运行完整的L1-L4预测分析"""
    if strategy_engine is None:
        error_detail = f"预测引擎未初始化"
        if strategy_engine_error:
            error_detail += f": {strategy_engine_error}"
        raise HTTPException(status_code=503, detail=error_detail)
    
    try:
        # 获取当前时间（纽约时区）
        ny_tz = pytz.timezone("America/New_York")
        target_dt = datetime.now(ny_tz).replace(tzinfo=None)
        
        print(f"开始预测分析 - 时间: {target_dt}")
        
        # 运行完整分析
        results = strategy_engine.analyze(target_dt)
        
        print(f"预测分析完成 - L1 Safe: {results.get('l1_safe')}")
        
        # 转换为JSON可序列化格式（确保所有类型都是Python原生类型）
        response = {
            "timestamp": target_dt.isoformat(),
            "l1_safe": bool(results.get('l1_safe', False)),  # 转换为Python bool
            "l1_prob": float(results.get('l1_prob', 0)),
            "l2_timestamp": results.get('l2_timestamp').isoformat() if results.get('l2_timestamp') else None,
            "l3_timestamp": results.get('l3_timestamp').isoformat() if results.get('l3_timestamp') else None,
        }
        
        # 缓存结果供其他端点使用
        state_manager._prediction_cache = results
        
        return response
    except Exception as e:
        import traceback
        print(f"❌ 预测失败: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get("/api/models/l1")
async def get_l1_debug():
    """获取L1市场择时详细信息"""
    try:
        cache = getattr(state_manager, '_prediction_cache', None)
        if not cache:
            raise HTTPException(status_code=404, detail="请先运行预测 /api/predict")
        
        return {
            "safe": bool(cache.get('l1_safe', False)),
            "probability": float(cache.get('l1_prob', 0)),
            "threshold": 0.5,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/l2")
async def get_l2_debug():
    """获取L2标的筛选详细信息"""
    try:
        cache = getattr(state_manager, '_prediction_cache', None)
        if not cache:
            raise HTTPException(status_code=404, detail="请先运行预测 /api/predict")
        
        l2_ranked = cache.get('l2_ranked')
        if l2_ranked is None or l2_ranked.empty:
            return {"stocks": []}
        
        # 转换为列表格式
        stocks = []
        for idx, row in l2_ranked.iterrows():
            stocks.append({
                "rank": int(idx + 1),
                "symbol": str(row['symbol']),
                "close": float(row['close']),
                "rank_score": float(row['rank_score']),
            })
        
        return {
            "timestamp": cache.get('l2_timestamp').isoformat() if cache.get('l2_timestamp') else None,
            "stocks": stocks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/l3")
async def get_l3_debug():
    """获取L3趋势确认详细信息"""
    try:
        cache = getattr(state_manager, '_prediction_cache', None)
        if not cache:
            raise HTTPException(status_code=404, detail="请先运行预测 /api/predict")
        
        l3_signals = cache.get('l3_signals')
        if l3_signals is None or l3_signals.empty:
            return {"signals": []}
        
        # 转换为列表格式
        signals = []
        for _, row in l3_signals.iterrows():
            shakeout_desc = "None"
            if row.get('shakeout_bull') == 1:
                shakeout_desc = "Bullish Shakeout"
            elif row.get('shakeout_bear') == 1:
                shakeout_desc = "Bearish Trap"
            
            signals.append({
                "symbol": str(row['symbol']),
                "close": float(row['close']),
                "long_p": float(row['long_p']),
                "short_p": float(row['short_p']),
                "shakeout": str(shakeout_desc),
            })
        
        return {
            "timestamp": cache.get('l3_timestamp').isoformat() if cache.get('l3_timestamp') else None,
            "signals": signals,
            "signal_threshold": float(SIGNAL_THRESHOLD)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/l4")
async def get_l4_debug():
    """获取L4风控建议详细信息"""
    if strategy_engine is None:
        raise HTTPException(status_code=503, detail="预测引擎未初始化")
    
    try:
        cache = getattr(state_manager, '_prediction_cache', None)
        if not cache:
            raise HTTPException(status_code=404, detail="请先运行预测 /api/predict")
        
        l2_ranked = cache.get('l2_ranked')
        l3_signals = cache.get('l3_signals')
        l1_safe = cache.get('l1_safe', False)
        
        if l2_ranked is None or l2_ranked.empty or l3_signals is None or l3_signals.empty:
            return {"long_recommendations": [], "short_recommendations": []}
        
        # 获取高置信度信号
        long_signals = strategy_engine.filter_signals(l3_signals, direction="long")
        short_signals = strategy_engine.filter_signals(l3_signals, direction="short")
        
        # 做多建议
        long_recs = []
        if l1_safe:
            for _, signal in long_signals.iterrows():
                predicted_return = strategy_engine.predict_return(signal['symbol'], l2_ranked)
                allocation = strategy_engine.get_allocation(signal['symbol'], l2_ranked, l1_safe)
                risk = strategy_engine.get_risk_params(signal['symbol'], "long", l2_ranked)
                
                if risk:
                    long_recs.append({
                        "symbol": str(signal['symbol']),
                        "confidence": float(signal['long_p']),
                        "predicted_return": float(predicted_return),
                        "allocation": float(allocation),
                        "current_price": float(signal['close']),
                        "take_profit": float(risk['take_profit']),
                        "stop_loss": float(risk['stop_loss']),
                        "tp_pct": float(risk['tp_pct']),
                        "sl_pct": float(risk['sl_pct']),
                        "risk_reward": float(risk['risk_reward']),
                    })
        
        # 做空建议
        short_recs = []
        for _, signal in short_signals.iterrows():
            predicted_return = strategy_engine.predict_return(signal['symbol'], l2_ranked)
            allocation = strategy_engine.get_allocation(signal['symbol'], l2_ranked, l1_safe)
            risk = strategy_engine.get_risk_params(signal['symbol'], "short", l2_ranked)
            
            if risk:
                short_recs.append({
                    "symbol": str(signal['symbol']),
                    "confidence": float(signal['short_p']),
                    "predicted_return": float(predicted_return),
                    "allocation": float(allocation),
                    "current_price": float(signal['close']),
                    "take_profit": float(risk['take_profit']),
                    "stop_loss": float(risk['stop_loss']),
                    "tp_pct": float(risk['tp_pct']),
                    "sl_pct": float(risk['sl_pct']),
                    "risk_reward": float(risk['risk_reward']),
                })
        
        return {
            "long_recommendations": long_recs,
            "short_recommendations": short_recs,
            "top_n": TOP_N_TRADES,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 静态文件服务 ====================

@app.get("/")
async def serve_root():
    """提供首页"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return {
            "message": "Dashboard 前端尚未创建",
            "hint": "请检查 web/static/index.html 是否存在"
        }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "algo-trade-dashboard"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
