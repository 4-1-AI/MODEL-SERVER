#main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from routers import step_router, websocket_router

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router.router, prefix="/ws")