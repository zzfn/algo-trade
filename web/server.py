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

from web.state_manager import StateManager

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
