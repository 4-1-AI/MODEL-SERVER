#time_series_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import os

class FireLevelDeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 7, 6) → (B, 6, 7)
        x = self.conv(x)
        return self.fc(x)

# 모델 로딩
model_path = "C:/Users/minju.MINJU-COM-1.000/Desktop/4-1/인공지능/플젝/aimodel/fire_step.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fire_step_model = FireLevelDeepCNN().to(device)
fire_step_model.load_state_dict(torch.load(model_path, map_location=device))
fire_step_model.eval()

# 특징 벡터 생성
def generate_feature(boxes: list, buffer: deque):
    flame_area = sum(b['w'] * b['h'] for b in boxes if b['label'] == '1')
    smoke_area = sum(b['w'] * b['h'] for b in boxes if b['label'] == '2')
    total_area = flame_area + smoke_area + 1e-6
    ratio = flame_area / total_area if total_area > 0 else 0

    if buffer:
        prev_flame, prev_smoke = buffer[-1][0], buffer[-1][1]
        delta_flame = flame_area - prev_flame
        delta_smoke = smoke_area - prev_smoke
        prev_ratio = prev_flame / (prev_flame + prev_smoke + 1e-6) if (prev_flame + prev_smoke) > 0 else 0
        delta_ratio = ratio - prev_ratio
    else:
        delta_flame = delta_smoke = delta_ratio = 0

    time_index = len(buffer) / 6

    return [
        np.log1p(flame_area),
        np.log1p(smoke_area),
        delta_flame,
        delta_smoke,
        delta_ratio,
        time_index
    ]

# 예측 함수
def predict_fire_stage(buffer: deque):
    if len(buffer) < 7:
        return -1

    X_seq = torch.tensor([list(buffer)], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = fire_step_model(X_seq)
        status = torch.argmax(F.softmax(outputs, dim=1), dim=1).item()
    return status
