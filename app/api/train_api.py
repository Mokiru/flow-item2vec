from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.schemas.request import TriggerTrainRequest
from app.schemas.response import TrainResponse, StatusResponse
from app.core.state import app_state
from app.core.task_manager import TrainingTaskManager

router = APIRouter(prefix="/api/v1/train", tags=["训练管理"])


@router.post("/trigger", response_model=TrainResponse)
async def trigger_training(req: TriggerTrainRequest):
    if app_state.is_training:
        raise HTTPException(status_code=409, detail="任务执行中")

    try:
        today = datetime.strptime(req.trainDate, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式错误，请使用 YYYY-MM-DD")

    # 🌟 提前预检：告诉调用方底层是不是挂了
    if app_state._init_error is not None and not app_state._is_initialized:
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail=f"依赖服务异常，无法启动训练: {app_state._init_error}"
        )

    success = TrainingTaskManager.start_training_if_idle(today)

    if not success:
        raise HTTPException(status_code=409, detail="任务执行中")

    return TrainResponse(code=200, message="已接收并后台执行")


@router.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        isTraining=app_state.is_training,
        message=app_state.status_message
    )