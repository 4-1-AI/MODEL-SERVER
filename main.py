from fastapi import FastAPI
from routers import predict, webrtc

app = FastAPI()

app.include_router(predict.router, prefix="/predict")
app.include_router(webrtc.router, prefix="/ws")