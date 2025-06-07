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

            if boxes:
                await process_fire_status(boxes, buffer, websocket)
                # 🔄 2. 이미지 → base64 인코딩
                
                try:
                    _, jpeg = cv2.imencode('.jpg', drawn_frame)
                    img_base64 = base64.b64encode(jpeg).decode("utf-8")
                    img_base64_str = f"data:image/jpeg;base64,{img_base64}"

                    # 🔄 3. 원인 분석은 비동기 태스크로 수행
                    asyncio.create_task(post_fire_cause(fire_center, img_base64_str))

                except Exception as e:
                    print("❌ drawn_frame 인코딩 실패:", e)
            else:
                # box 없을 때도 상태 처리
                await process_fire_status([], buffer, websocket)
    
    except WebSocketDisconnect:
        print("🔌 WebSocket 연결 종료")
