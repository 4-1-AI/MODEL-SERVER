from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
import cv2, numpy as np
from collections import deque
import asyncio

from my_models.yolov5_model import detect_fire_v5
from routers.step_router import process_fire_status
from routers.predict_router import post_fire_cause

router = APIRouter()

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = deque(maxlen=7)

    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                print("⚠️ 잘못된 프레임 수신 - 건너뜀")
                continue
            print(f"[RECV] 프레임 수신: {frame.shape}")

            # 🔄 YOLO 감지만 수행
            try:
                fire_center, boxes, drawn_frame = detect_fire_v5(frame)
                print(f"[YOLO] 예측 결과: {len(boxes)}개")
            except Exception as e:
                print("🔥 detect_fire_v5 예외:", e)
                continue

            for box in boxes:
                print(f" → label: {box['label']}, conf: {box['confidence']:.2f}")

            # 🔁 실시간 상태 전송 (이미지 없음)
            await websocket.send_json({
                "type": "status",
                "boxes": boxes,
                "status": -1,  # 기본값 또는 추후 모델로 예측
                "statusLabel": "safe"  # 추후 감지된 상태 반영 가능
            })

            # 🔀 추가 로직 (비동기 실행)
            if boxes:
                asyncio.create_task(process_fire_status(boxes, buffer, websocket))
                asyncio.create_task(post_fire_cause(fire_center, None))

    except WebSocketDisconnect:
        print("🔌 WebSocket 연결 종료")
