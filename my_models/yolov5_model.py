# yolov5_model.py
import sys
import os
from PIL import Image 
import cv2

# 현재 파일 기준으로 yolov5 디렉토리 경로 설정
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov5")
sys.path.append(YOLOV5_PATH)

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
    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # transform: Resize + ToTensor
    img_tensor = transform(pil_image).unsqueeze(0)    

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
        return (cx, cy), pred, label #bbox 중심 좌표, bbox 전체 좌표, 클래스 이름(smoke / fire)
    return None, None, None