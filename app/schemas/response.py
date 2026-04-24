from pydantic import BaseModel, Field

class TrainResponse(BaseModel):
    """触发训练接口的响应体"""
    code: int = Field(..., description="状态码，200表示接受任务，409表示冲突")
    message: str = Field(..., description="状态描述信息")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "code": 200,
                    "message": "已接收并后台执行"
                }
            ]
        }

class StatusResponse(BaseModel):
    """查询训练状态的响应体"""
    isTraining: bool = Field(..., description="当前是否正在训练")
    message: str = Field(..., description="当前状态详情或最新日志")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "isTraining": True,
                    "message": "训练进行中..."
                }
            ]
        }

class HealthResponse(BaseModel):
    """健康检查响应体"""
    status: str = Field(default="UP", description="服务状态")