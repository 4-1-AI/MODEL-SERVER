import sys, os
import torch
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

# 1. yolov5 경로 연결
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov5")
sys.path.append(YOLOV5_PATH)

# 2. yolov5 내부 모듈 import
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# 3. 모델 로딩
device = select_device('')
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fire_smoke.pt"))
model = attempt_load(model_path,  device=device)
model.eval()

def enhance_fire_smoke_image(image):
    """
    불꽃과 연기 감지를 동시에 강화하기 위한 전처리 (톤 왜곡 최소화):
    - 대비/밝기/채도 증가를 최소한으로 조정
    - CLAHE는 유지 (연기 감지에 효과적)
    - 선명화는 약하게 적용
    """

    # 1. 대비 & 밝기 (톤 왜곡 방지용 최소값)
    alpha = 1.1  # 기존보다 약하게
    beta = 5
    img_enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 2. 채도/명도 약간만 증가
    hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = cv2.add(s, 10)  # 기존 40 → 10
    v = cv2.add(v, 5)   # 기존 15 → 5
    hsv = cv2.merge([h, s, v])
    img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. CLAHE (밝은 영역 왜곡 없도록 clipLimit 낮춤)
    lab = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # 기존 2.0 → 1.5
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # 4. 선명화 (비율 줄임)
    blurred = cv2.GaussianBlur(img_enhanced, (0, 0), sigmaX=1.0)  # 기존 1.5 → 1.0
    sharpened = cv2.addWeighted(img_enhanced, 1.2, blurred, -0.2, 0)  # 기존 1.5:-0.5 → 1.2:-0.2

    return sharpened


def detect_fire_v5(image, label_keywords=["fire", "smoke"]):
    # 0. 해상도 먼저 보정 (최소 640x480 보장)
    if image.shape[0] < 480 or image.shape[1] < 640:
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)

    # # 0. 전처리로 감지 강화
    enhanced = enhance_fire_smoke_image(image)

    # 1. BGR → RGB → PIL
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # 2. transform 후 추론
    img_tensor = transform(pil_image).unsqueeze(0)
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)[0]
    
    boxes = []
    fire_center = None
    
    if pred is not None and len(pred):
        for *xyxy, conf, cls in pred:
            label = model.names[int(cls)]
            if label not in label_keywords:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            fire_center = (cx, cy)
            confidence = float(conf)

            label_id = str(int(cls))  

            boxes.append({
                "label": label_id,
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "confidence": confidence
            })

    return fire_center, boxes, image