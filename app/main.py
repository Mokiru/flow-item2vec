from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.train_api import router as train_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：你的 datasource 里的预拉取线程会自动在 __init__ 里启动
    print("Item2Vec Service Started.")
    yield
    # 关闭时清理
    print("Item2Vec Service Shutting down.")

app = FastAPI(title="Item2Vec Service", lifespan=lifespan)

# 注册路由
app.include_router(train_router)

@app.get("/health")
async def health():
    return {"status": "UP"}