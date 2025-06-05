import sys
import os

# 현재 파일 기준으로 yolov5 디렉토리 경로 설정
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov5")
sys.path.append(YOLOV5_PATH)

import sys
print("\n".join(sys.path))


from models.experimental import attempt_load

import torch
from torchvision import transforms
from utils.general import non_max_suppression

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fire_smoke.pt"))
model = attempt_load(model_path, device="cpu")

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

def detect_fire_v5(image, label_keywords=["fire", "smoke"]):
    img_tensor = transform(image).unsqueeze(0)
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)[0]
    if len(pred) == 0:
        return None, None, None

    for *xyxy, conf, cls in pred:
        label = model.names[int(cls)]
        if label not in label_keywords:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return (cx, cy), pred, label
    return None, None, None