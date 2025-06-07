from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
import cv2, numpy as np
from collections import deque
import asyncio
import base64

from my_models.yolov5_model import detect_fire_v5
from routers.step_router import process_fire_status
from routers.predict_router import post_fire_cause

router = APIRouter()

# WebSocket별 사용자 ID 저장소
active_users = {}
already_sent_users = set()
# 사용자별로 감지 연속 카운터
detection_counter = {}
# 사용자별로 안전(safe) 상태 연속 카운터 추가
safe_counter = {}

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    user_id_msg = await websocket.receive_text()
    user_id = int(user_id_msg)

    active_users[websocket] = user_id
    print(f"✅ 연결된 사용자 ID: {user_id}")
    
    buffer = deque(maxlen=7)

    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                print("⚠️ 잘못된 프레임 수신 - 건너뜀")
                continue
            # print(f"[RECV] 프레임 수신: {frame.shape}")

            try:
                fire_center, boxes, drawn_frame = detect_fire_v5(frame)
            except Exception as e:
                print("🔥 detect_fire_v5 예외:", e)
                continue

            if boxes:
                detection_counter[user_id] = detection_counter.get(user_id, 0) + 1
                await process_fire_status(boxes, buffer, websocket)
                
                # 5프레임 연속 감지되었고 아직 문자 전송하지 않은 경우
                if detection_counter[user_id] >= 5 and user_id not in already_sent_users:
                    already_sent_users.add(user_id)
                    
                    try:
                        _, jpeg = cv2.imencode('.jpg', drawn_frame)
                        img_base64 = base64.b64encode(jpeg).decode("utf-8")
                        img_base64_str = f"data:image/jpeg;base64,{img_base64}"
                        asyncio.create_task(post_fire_cause(fire_center, img_base64_str, user_id))
                        print("📨 문자 전송 완료")
                    except Exception as e:
                        print("❌ 인코딩 실패:", e)

            else:
                detection_counter[user_id] = 0
                await process_fire_status([], buffer, websocket)
                # 안전 상태 프레임 카운터 증가
                safe_counter[user_id] = safe_counter.get(user_id, 0) + 1
                
                # 20프레임 이상 연속 safe 상태이면 모든 정보 초기화
                if safe_counter[user_id] >= 20:
                    print(f"🧯 사용자 {user_id}: 20프레임 연속 safe → 상태 초기화")
                    active_users.pop(websocket, None)
                    already_sent_users.discard(user_id)
                    detection_counter.pop(user_id, None)
                    safe_counter.pop(user_id, None)
    
    except WebSocketDisconnect:
        print("🔌 WebSocket 연결 종료")
        # 해당 유저 설정 정보 모두 초기화
        active_users.pop(websocket, None)
        already_sent_users.discard(user_id)
        detection_counter.pop(user_id, None)
