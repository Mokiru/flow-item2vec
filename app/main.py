from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Service Started.")
    yield
    # 关闭时清理
    print("Service Shutting down.")

app = FastAPI(title="Item2Vec Service", lifespan=lifespan)

# 注册路由
# app.include_router(train_router)

@app.get("/health")
async def health():
    return {"status": "UP"}