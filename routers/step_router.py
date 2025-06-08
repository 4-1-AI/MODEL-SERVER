from fastapi import APIRouter, WebSocket
import torch
from collections import deque
from my_models.time_series_model import generate_feature, fire_step_model, device
from collections import Counter

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
        return -1, "safe"

    # 2. 감지된 프레임일 때만 feature 생성 및 buffer 저장
    feature = generate_feature(boxes, buffer)
    buffer.append(feature)

    # 3. 연속 감지된 7개가 모였을 때만 판단
    statuses = []

    # 상태 예측을 계속하면서 누적
    if len(buffer) >= 1:
        for i in range(len(buffer)):
            with torch.no_grad():
                tensor = torch.tensor([list([buffer[i]])], dtype=torch.float32).to(device)
                status = torch.argmax(fire_step_model(tensor), dim=1).item()
            
            statuses.append(status)

        # 7개 이상의 상태가 누적되었을 경우 가장 많이 등장한 상태 선택
        if len(statuses) >= 7:
            most_common_status = Counter(statuses).most_common(1)[0][0]

            # 상태 라벨 변환
            label = (
                "caution" if most_common_status == 0 else
                "danger" if most_common_status == 1 else
                "danger" if most_common_status == 2 else
                "unknown"
            )

            # 가장 많이 등장한 상태 반환
            await websocket.send_json({
                "type": "status",
                "status": most_common_status,  # 가장 많이 감지된 상태
                "statusLabel": label  # 해당 상태의 레이블
            })
            
                # `caution` 또는 `danger` 상태일 경우만 반환
        if status == 0:  # `caution` 상태
            return status, "caution"
        elif status == 1 or status == 2:  # `danger` 상태
            return status, "danger"
        else:
            # 다른 상태는 처리하지 않음
            return None, None