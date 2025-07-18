import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import torch
import torch.nn as nn
from collections import deque
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
import os
import yaml

cwd = os.getcwd()
config_path = os.path.join(cwd,"../../../","PsychologyBot" ,"Config", "config.yaml")
config = yaml.safe_load(open(config_path))
model_dir_config=cwd+"/../../.."+config['CommonConfig']['ModelDir']
processed_data_dir=cwd+"/../../.."+config['CommonConfig']['ProcessedData']
DATA_DIR = processed_data_dir+"features/"
MODEL_DIR=model_dir_config
LABEL_ENCODER_PATH=MODEL_DIR+"label_encoding.pkl"
MODEL_WEIGHTS=MODEL_DIR+"best_model.pth"

# ========= Flask App =========
app = Flask(__name__)

# ========= Load Label Map =========
label_encoder = joblib.load(LABEL_ENCODER_PATH)
label_decoder = {v: k for k, v in label_encoder.items()}

class GRUClassifier(nn.Module):
    def __init__(self, input_size=126, hidden_size=64, num_classes=151):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.3)
        self.gru2 = nn.GRU(hidden_size, 128, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.3)
        self.gru3 = nn.GRU(128, 64, batch_first=True, bidirectional=False)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 256)
        self.drop3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop4 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):  # x: (B, T, V, C)
        B, T, V, C = x.shape
        x = x.view(B, T, V * C)  # (B, 50, 126)
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x, _ = self.gru3(x)  # (B, 50, 64)
        x = x[:, -1, :]      # use last time step (B, 64)
        x = self.fc1(x)
        x = self.drop3(x)
        x = self.fc2(x)
        x = self.drop4(x)
        x = self.fc3(x)
        return self.out(x)

# ========= Load Model =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier(input_size=126, num_classes=len(label_encoder))
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.to(device)
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.9)

BUFFER_SIZE = 50
landmark_buffer = deque(maxlen=BUFFER_SIZE)
last_pred_label = "None"

def normalize_hand_kps_zscore(hand_kps):
    if np.all(hand_kps == 0):
        return hand_kps

    # Remove wrist (origin shift)
    centered = hand_kps - hand_kps[0]

    # Z-score normalize each channel (x, y, z) separately
    mean = centered.mean(axis=0)        # (3,)
    std = centered.std(axis=0) + 1e-6   # avoid division by 0
    normalized = (centered - mean) / std
    return normalized

def extract_full_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            if label == 'Left':
                left_hand = normalize_hand_kps_zscore(keypoints)
            else:
                right_hand = normalize_hand_kps_zscore(keypoints)
    return np.concatenate([left_hand, right_hand], axis=0)  # (42, 3)

def predict_from_frames(frames):  # frames = list of 50 RGB frames
    buffer = []
    for frame in frames:
        landmark = extract_full_landmarks(frame)
        buffer.append(landmark)

    input_data = np.array(buffer, dtype=np.float32)  # (50, 69, 5)
    input_tensor = torch.tensor(input_data).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        label = label_decoder[pred_idx]
    return label