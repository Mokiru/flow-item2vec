from pydantic import BaseModel, Field

class TriggerTrainRequest(BaseModel):
    """触发训练的请求体"""
    trainDate: str = Field(..., description="训练日期，格式 YYYY-MM-DD")