import cv2
import mediapipe as mp
import numpy as np
import os
import random
from tqdm import tqdm
from multiprocessing import Pool
import yaml

mp_hands = mp.solutions.hands
cwd = os.getcwd()
config_path = os.path.join(cwd,"PsychologyBot" ,"Config", "config.yaml")
config = yaml.safe_load(open(config_path))
data_dir=cwd+config['CommonConfig']['DataDir']
processed_data_dir=cwd+config['CommonConfig']['ProcessedData']
VIDEO_DIR = data_dir+"generated_videos/"
HAND_DIMS = 21 * 3 * 2
MAX_FRAMES = 45
AUGMENTATIONS = ['horizontal_flip_landmarks', 'brightness', 'contrast', 'saturation', 'grayscale','small_rotation','scale','translate']
MAX_POOL_WORKERS = 25
SKIP = ['even', 'odd']
SPEED_COUNT = 15
ORIGINAL_COUNT=1
mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=True)

def normalize_hand_kps_zscore(hand_kps):
    if np.all(hand_kps == 0):
        return hand_kps
    centered = hand_kps - hand_kps[0]
    mean = centered.mean(axis=0)  
    std = centered.std(axis=0) + 1e-6  
    normalized = (centered - mean) / std
    return normalized

def create_temp_speed_video(original_path, output_path, speed=1.0):
    cap = cv2.VideoCapture(original_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if speed < 1.0:
        factor = int(1 / speed)
        frames = [f for f in frames for _ in range(factor)]
    elif speed > 1.0:
        frames = frames[::int(speed)]

    for f in frames:
        out.write(f)
    out.release()

def apply_augmentation(frame, aug_type):
    if aug_type == 'horizontal_flip_landmarks':
        flipped_frame = cv2.flip(frame, 1)
        frame=flipped_frame
    elif aug_type == 'flip':
        frame = cv2.flip(frame, 1)
    elif aug_type == 'brightness':
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=random.randint(30, 60))
    elif aug_type == 'contrast':
        frame = cv2.convertScaleAbs(frame, alpha=random.uniform(1.5, 2.0), beta=0)
    elif aug_type == 'saturation':
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(1.5, 2.0), 0, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif aug_type == 'grayscale':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif aug_type == 'jitter':
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

    elif aug_type == 'small_rotation':
        angle = random.uniform(-10, 10)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    elif aug_type == 'scale':
        scale = random.uniform(0.9, 1.1)
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # crop or pad to original size
        dh, dw = resized.shape[:2]
        top = max((dh - h) // 2, 0)
        left = max((dw - w) // 2, 0)
        frame = resized[top:top + h, left:left + w]

    elif aug_type == 'translate':
        tx = random.randint(-10, 10)
        ty = random.randint(-10, 10)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    return frame

def get_key_points(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = mp_holistic.process(rgb)

    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))

    if hand_results.left_hand_landmarks:
        hand_kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_results.left_hand_landmarks.landmark])
        left_hand = normalize_hand_kps_zscore(hand_kp)
        
    if hand_results.right_hand_landmarks:
        hand_kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_results.right_hand_landmarks.landmark])
        right_hand = normalize_hand_kps_zscore(hand_kp)

    return np.concatenate([left_hand, right_hand], axis=0)

def extract_hand_keypoints(video_path, orignal=None, aug_type=None, skip_pattern=None):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if skip_pattern == "even" and frame_idx % 2 == 0:
            frame_idx += 1
            continue
        elif skip_pattern == "odd" and frame_idx % 2 != 0:
            frame_idx += 1
            continue

        if aug_type:
            frame = apply_augmentation(frame, aug_type)

        keypoints = get_key_points(frame)
        all_keypoints.append(keypoints)
        frame_idx += 1

    cap.release()

    if len(all_keypoints) == 0:
        return np.zeros((MAX_FRAMES, 42, 3))

    if len(all_keypoints) < MAX_FRAMES:
        repeats = (MAX_FRAMES + len(all_keypoints) - 1) // len(all_keypoints)
        all_keypoints = (all_keypoints * repeats)[:MAX_FRAMES]
    else:
        all_keypoints = all_keypoints[:MAX_FRAMES]

    return np.array(all_keypoints)

def process_video(file, input_dir, output_dir):
    if not file.endswith(('.mp4', '.mov')):
        return

    base_label = os.path.splitext(file)[0]
    video_path = os.path.join(input_dir, file)

    for i in range(ORIGINAL_COUNT):
        features = extract_hand_keypoints(video_path, orignal="original")
        np.save(os.path.join(output_dir, f"{base_label}_orig{i:02d}.npy"), features)

    for aug in AUGMENTATIONS:
        aug_features = extract_hand_keypoints(video_path, aug_type=aug)
        np.save(os.path.join(output_dir, f"{base_label}_aug_{aug}.npy"), aug_features)

    for skp in SKIP:
        skp_features = extract_hand_keypoints(video_path, skip_pattern=skp)
        np.save(os.path.join(output_dir, f"{base_label}_skp_{skp}.npy"), skp_features)

    for aug in AUGMENTATIONS:
        for skp in SKIP:
            features = extract_hand_keypoints(video_path, aug_type=aug, skip_pattern=skp)
            np.save(os.path.join(output_dir, f"{base_label}_aug{aug[:2]}_skp{skp[0]}.npy"), features)

    for i in range(SPEED_COUNT):
        slow = random.uniform(0.3, 0.8)
        fast = random.uniform(1.5, 2.5)
        for speed, tag in [(slow, 'slow'), (fast, 'fast')]:
            temp_file = f"temp_{tag}_{i}_{base_label}.mp4"
            create_temp_speed_video(video_path, temp_file, speed=speed)
            features = extract_hand_keypoints(temp_file)
            np.save(os.path.join(output_dir, f"{base_label}_{tag}{i:02d}.npy"), features)
            os.remove(temp_file)

    for i in range(SPEED_COUNT):
        for aug in AUGMENTATIONS:
            for skp in SKIP:
                speed = random.uniform(0.3, 0.8) if random.random() < 0.5 else random.uniform(1.5, 2.5)
                temp_file = f"temp_augspd_{i}_{base_label}.mp4"
                create_temp_speed_video(video_path, temp_file, speed=speed)
                features = extract_hand_keypoints(temp_file, aug_type=aug, skip_pattern=skp)
                np.save(os.path.join(output_dir, f"{base_label}_aug{aug[:2]}_skp{skp[0]}_spd{i:02d}.npy"), features)
                os.remove(temp_file)

def process_all(input_dir=VIDEO_DIR, output_dir=processed_data_dir+'features'):
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    with Pool(processes=MAX_POOL_WORKERS) as pool:
        async_results = [pool.apply_async(process_video, (file, input_dir, output_dir)) for file in files]
        for result in async_results:
            result.get()

if __name__ == "__main__":
    process_all()
