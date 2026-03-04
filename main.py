"""
COCO Vision Metrology — FastAPI Entry Point

啟動時預載所有模型 (Mask R-CNN, HRNet ONNX, SAM)。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.metrology import router as metrology_router
from app.core.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: 啟動時載入模型，關閉時釋放。"""
    print("=" * 60)
    print(" 🚀 COCO Vision Metrology — 啟動中")
    print("=" * 60)

    # 啟動: 檢查 checkpoint + 載入模型
    model_manager.ensure_checkpoints()
    model_manager.load_all()

    print("=" * 60)
    print(" ✅ API 就緒: http://localhost:8000/docs")
    print("=" * 60)

    yield  # 運行中

    # 關閉: 清理 (可選)
    print("\n🛑 關閉中 ...")


app = FastAPI(
    title="COCO Vision Metrology",
    description="動物眼睛偵測與距離量測 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(metrology_router, prefix="/v1/metrology", tags=["Metrology"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "maskrcnn": model_manager.maskrcnn is not None,
            "hrnet_onnx": model_manager.hrnet_session is not None,
            "sam": model_manager.sam_predictor is not None,
        },
    }
