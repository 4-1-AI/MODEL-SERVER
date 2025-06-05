from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
import cv2, base64, numpy as np
from collections import deque
import asyncio
from PIL import Image 

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

            # 1. YOLO 감지
            fire_center, pred, label = detect_fire_v5(frame)
            boxes = []

            # bbox 시각화화
            if pred is not None:
                for *xyxy, conf, cls in pred:
                    label_str = str(int(cls))
                    if label_str not in ['1', '2']:
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    confidence = float(conf)

                    boxes.append({
                        "label": label_str,
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "confidence": confidence
                    })

                    color = (0, 0, 255) if label_str == '1' else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 2. 시각화 프레임 전송
            _, encoded = cv2.imencode(".jpg", frame)
            img_base64 = base64.b64encode(encoded).decode("utf-8")

            await websocket.send_json({
                "image": img_base64,
                "boxes": boxes,
                "status": -1,
                "statusLabel": "safe"
            })

            # 3. bbox가 감지되었을 때 1-1, 1-2 병렬 처리
            if boxes:
                await asyncio.gather(
                    process_fire_status(boxes, buffer, websocket),
                    post_fire_cause(fire_center, img_base64)
                )

    except WebSocketDisconnect:
        print("🔌 WebSocket 연결 종료")
