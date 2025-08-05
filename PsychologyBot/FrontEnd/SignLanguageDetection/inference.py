import cv2
import mediapipe as mp
import numpy as np
import joblib
import torch
import torch.nn as nn
from collections import deque
from flask import Flask
import os
import yaml
from model import GRUClassifier

cwd = os.getcwd()
config_path = os.path.join(cwd,"../../../","PsychologyBot" ,"Config", "config.yaml")
config = yaml.safe_load(open(config_path))
model_dir_config=cwd+"/../../.."+config['CommonConfig']['ModelDir']
processed_data_dir=cwd+"/../../.."+config['CommonConfig']['ProcessedData']
DATA_DIR = processed_data_dir+"features/"
MODEL_DIR=model_dir_config
LABEL_ENCODER_PATH=MODEL_DIR+"label_encoding.pkl"
MODEL_WEIGHTS=MODEL_DIR+"best_model.pth"


app = Flask(__name__)


label_encoder = joblib.load(LABEL_ENCODER_PATH)
label_decoder = {v: k for k, v in label_encoder.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier(input_size=126, num_classes=len(label_encoder))
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.to(device)
model.eval()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

BUFFER_SIZE = 50
landmark_buffer = deque(maxlen=BUFFER_SIZE)
last_pred_label = "None"

def normalize_hand_kps(hand_kps):
    if np.all(hand_kps == 0):
        return hand_kps
    origin = hand_kps[0]
    centered = hand_kps - origin
    max_dist = np.linalg.norm(centered, axis=1).max()
    return centered / (max_dist + 1e-6)

def extract_full_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for i in range(len(hand_results.multi_hand_landmarks)):
            label = hand_results.multi_handedness[i].classification[0].label
            hand_kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_results.multi_hand_landmarks[i].landmark])
            if label.lower() == 'left':
                left_hand = normalize_hand_kps(hand_kp)
            elif label.lower() == 'right':
                right_hand = normalize_hand_kps(hand_kp)

    return np.concatenate([left_hand, right_hand], axis=0)

def predict_from_frames(frames): 
    buffer = []
    for frame in frames:
        landmark = extract_full_landmarks(frame)
        buffer.append(landmark)

    input_data = np.array(buffer, dtype=np.float32)
    input_tensor = torch.tensor(input_data).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        label = label_decoder[pred_idx]
    return label