from fastapi import APIRouter, WebSocket
import torch
from collections import deque
from my_models.time_series_model import generate_feature, fire_step_model, device

router = APIRouter()

async def process_fire_status(boxes, buffer: deque, websocket: WebSocket):
    if not boxes:
        await websocket.send_json({
            "type": "status",
            "boxes": [],
            "status": -1,
            "statusLabel": "safe"
        })
        buffer.clear()
        return 

    # 2. 감지된 프레임일 때만 feature 생성 및 buffer 저장
    feature = generate_feature(boxes, buffer)
    buffer.append(feature)

    # 3. 연속 감지된 7개가 모였을 때만 판단
    if len(buffer) == 7:
        with torch.no_grad():
            tensor = torch.tensor([list(buffer)], dtype=torch.float32).to(device)
            status = torch.argmax(fire_step_model(tensor), dim=1).item()
        
        label = (
            "caution" if status == 0 else
            "danger" if status == 1 else
            "danger" if status == 2 else
            "unknown"
        )

        await websocket.send_json({
            "type": "status",
            "status": status,
            "statusLabel": label
        })