import os
import cv2
import mediapipe as mp
import yaml

mp_hands = mp.solutions.hands

def has_hand_landmarks(image, hands_detector):
    results = hands_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results.multi_hand_landmarks is not None

def process_video_file(video_path, output_path, max_count=51, fps=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    valid_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= 5:  # Skip first 5 frames
            if has_hand_landmarks(frame, hands_detector):
                valid_frames.append(frame)
        frame_idx += 1

    cap.release()
    hands_detector.close()

    if not valid_frames:
        print(f"No hand landmark frames in {video_path}")
        return

    # Adjust frame count
    num_valid = len(valid_frames)
    if num_valid < max_count:
        last_frame = valid_frames[-1]
        valid_frames += [last_frame] * (max_count - num_valid)
    elif num_valid > max_count:
        start = (num_valid - max_count) // 2
        valid_frames = valid_frames[start:start + max_count]

    h, w, _ = valid_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in valid_frames:
        out.write(f)
    out.release()
    print(f"Saved: {output_path} ({len(valid_frames)} frames)")

def process_all_videos(input_dir, output_dir, max_count=51):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(('.mp4', '.mov', '.avi')):
            video_path = os.path.join(input_dir, file)
            label = os.path.splitext(file)[0]
            output_path = os.path.join(output_dir, f"{label}.mp4")
            process_video_file(video_path, output_path, max_count=max_count)

# === CONFIGURATION ===
cwd = os.getcwd()
config_path = os.path.join(cwd,"PsychologyBot" ,"Config", "config.yaml")
config = yaml.safe_load(open(config_path))
datadir=cwd+config['CommonConfig']['DataDir']
input_video_dir = datadir+"AlphabetDataset/INDIAN SIGN LANGUAGE ANIMATED VIDEOS /"    # Folder with input video files
output_video_dir = datadir+"generated_videos/"  # Output folder
max_count = 45                                # Fixed frame count per video
# =====================

process_all_videos(input_video_dir, output_video_dir, max_count)
