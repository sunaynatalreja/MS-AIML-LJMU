import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import mediapipe as mp
import yaml

# Transformation pipeline including augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Random crop to 224x224
    transforms.RandomHorizontalFlip(),   # Flip horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color jitter
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (use ImageNet stats)
])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def detect_hand_gesture(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return results.multi_hand_landmarks is not None

def get_motion_frames(frames, threshold=10):
    motion_frames = []
    prev_frame = None
    for frame in frames:
        if prev_frame is not None:
            diff = cv2.absdiff(frame, prev_frame)
            if np.mean(diff) > threshold:  # Include frame if motion is above the threshold
                motion_frames.append(frame)
        prev_frame = frame
    return np.array(motion_frames)

def sample_high_motion_frames(frames, num_samples):
    motion_scores = [np.mean(cv2.absdiff(frames[i], frames[i-1])) for i in range(1, len(frames))]
    top_indices = np.argsort(motion_scores)[-num_samples:]
    frame_data=[]
    for indices in top_indices:
        frame_data.append(frames[indices])
    return frame_data

def sample_frames_from_video(video_path, resize_dim=(224, 224), num_samples=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    prev_frame = None

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect meaningful gestures
        if detect_hand_gesture(frame):
            # Resize frame
            frame = cv2.resize(frame, resize_dim)
            # Add only frames with significant motion
            #if prev_frame is None or np.mean(cv2.absdiff(frame, prev_frame)) > 10:  # Motion threshold
            frames.append(frame)

    cap.release()

    # Sample top motion frames if necessary
    if len(frames) > num_samples:
        frames = sample_high_motion_frames(frames, num_samples)

    return np.array(frames)

def pad_or_truncate_frames(frames, target_frame_count):
    """Pads or truncates the frames to ensure consistent frame count."""
    current_frame_count = len(frames)
    
    if current_frame_count < target_frame_count:
        # If there are fewer frames, pad with duplicates (or you could use zeros or other padding strategies)
        padding = target_frame_count - current_frame_count
        # Pad along the first axis (frames axis)
        frames = np.pad(frames, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='edge')
    elif current_frame_count > target_frame_count:
        # If there are more frames, truncate to the target count
        frames = frames[:target_frame_count]
    return frames

def augment_frames(frames):
    augmented_frames = []
    
    # Apply one augmentation per frame
    for frame in frames:
        # Randomly choose one augmentation for each frame
        choice = np.random.choice([
            'flip', 'rotate', 'noise', 'blur', 
            'brightness', 'contrast', 'saturation', 
            'crop', 'shear', 'grayscale', 
            'hue', 'cutout'
        ])
        
        if choice == 'flip':
            augmented_frame = np.flip(frame, axis=1)  # Horizontal flip
        elif choice == 'rotate':
            angle = np.random.randint(0, 360)
            augmented_frame = np.rot90(frame, k=angle // 90)  # Rotate by multiples of 90 degrees
        elif choice == 'noise':
            augmented_frame = np.clip(frame + np.random.randint(0, 50, frame.shape, dtype=np.int32), 0, 255).astype(np.uint8)
        elif choice == 'blur':
            augmented_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Gaussian blur
        elif choice == 'brightness':
            factor = np.random.uniform(0.8, 1.2)
            augmented_frame = np.clip(frame * factor, 0, 255).astype(np.uint8)  # Adjust brightness
        elif choice == 'contrast':
            factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(frame, axis=(0, 1), keepdims=True)
            augmented_frame = np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)  # Adjust contrast
        elif choice == 'saturation':
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_frame[:, :, 0] = ((hsv_frame[:, :, 0].astype(int) + np.random.randint(-10, 10)) % 180).astype(np.uint8)
            augmented_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)  # Adjust saturation
        elif choice == 'crop':
            h, w, _ = frame.shape
            crop_x = np.random.randint(0, int(0.2 * w))
            crop_y = np.random.randint(0, int(0.2 * h))
            augmented_frame = frame[crop_y:h-crop_y, crop_x:w-crop_x]
            augmented_frame = cv2.resize(augmented_frame, (w, h))  # Crop and resize back
        elif choice == 'shear':
            shear_factor = np.random.uniform(-0.2, 0.2)
            h, w, _ = frame.shape
            M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
            augmented_frame = cv2.warpAffine(frame, M, (w, h))
        elif choice == 'grayscale':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            augmented_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel
        elif choice == 'hue':
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            augmented_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)  # Adjust hue
        elif choice == 'cutout':
            h, w, _ = frame.shape
            cutout_size = int(0.2 * min(h, w))
            x1 = np.random.randint(0, w - cutout_size)
            y1 = np.random.randint(0, h - cutout_size)
            augmented_frame = frame.copy()
            augmented_frame[y1:y1+cutout_size, x1:x1+cutout_size] = 0  # Blackout a region
        # elif choice == 'scale':
        #     scale_factor = np.random.uniform(0.8, 1.2)
        #     augmented_frame=cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        # scale_factor = np.random.uniform(0.8, 1.2)
        # augmented_frame=np.array([cv2.resize(frame, None, fx=scale_factor, fy=scale_factor) for frame in frames])
        augmented_frames.append(augmented_frame)
    
    return np.array(augmented_frames)

def ExtractAlphabeticalDataset(video_folder, resize_dim=(224, 224), augment_count=3):
    data = []
    labels = []
    max_frame_count = 0  # Variable to track the maximum frame count across all videos
    
    # Get all video files in the directory
    videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
    # First pass to find the max frame count (after trimming based on hand gestures)
    for video in videos:
        video_path = os.path.join(video_folder, video)
        frames = sample_frames_from_video(video_path, resize_dim)
        max_frame_count = max(max_frame_count, len(frames))  # Update the maximum frame count
    
    # Second pass to process the videos
    for video in videos:
        class_label = video.split('.')[0]  # Assuming filename format is: class_label_video.mp4
        video_path = os.path.join(video_folder, video)
        
        # Sample frames from the video with trimming based on hand gestures
        frames = sample_frames_from_video(video_path, resize_dim)
        frames = pad_or_truncate_frames(frames, max_frame_count)  # Pad frames to the max frame count
        transformed_frames=[]
        # Apply transformations to frames
        for frame in frames:
            transformed_frames.append(transform(Image.fromarray(frame.astype(np.uint8))))
        
        # Stack the transformed frames into a tensor (for input to X3D)
        data.append(torch.stack(transformed_frames, dim=1))
        labels.append(class_label)
        
        # Augment the frames and apply transformations to augmented data
        augmented_data = []
        for i in range(1,augment_count):
            augmented_data=augment_frames(frames)
            augmented_data=pad_or_truncate_frames(augmented_data, max_frame_count)
            aug_transformed=[]
            for aug in augmented_data:
                aug_transformed.append(transform(Image.fromarray(aug.astype(np.uint8))))  
            data.append(torch.stack(aug_transformed, dim=1))
            labels.append(class_label)
    return torch.stack(data), np.array(labels) 

def ExtractCategoricalDataset(video_folder, resize_dim=(224, 224), augment_count=25):
    data = []
    labels = []
    max_frame_count = 0  # Variable to track the maximum frame count across all videos
    
    # Get all video files in the directory
    videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
    # First pass to find the max frame count (after trimming based on hand gestures)
    for video in videos:
        video_path = os.path.join(video_folder, video)
        frames = sample_frames_from_video(video_path, resize_dim)
        max_frame_count = max(max_frame_count, len(frames))  # Update the maximum frame count
    
    # Second pass to process the videos
    for video in videos:
        class_label = video.split('.')[0]  # Assuming filename format is: class_label_video.mp4
        video_path = os.path.join(video_folder, video)
        
        # Sample frames from the video with trimming based on hand gestures
        frames = sample_frames_from_video(video_path, resize_dim)
        frames = pad_or_truncate_frames(frames, max_frame_count)  # Pad frames to the max frame count
        transformed_frames=[]
        # Apply transformations to frames
        for frame in frames:
            transformed_frames.append(transform(Image.fromarray(frame.astype(np.uint8))))
        
        # Stack the transformed frames into a tensor (for input to X3D)
        data.append(torch.stack(transformed_frames, dim=1))
        labels.append(class_label)
        
        # Augment the frames and apply transformations to augmented data
        augmented_data = []
        for i in range(1,augment_count):
            augmented_data=augment_frames(frames)
            augmented_data=pad_or_truncate_frames(augmented_data, max_frame_count)
            aug_transformed=[]
            for aug in augmented_data:
                aug_transformed.append(transform(Image.fromarray(aug.astype(np.uint8))))  
            data.append(torch.stack(aug_transformed, dim=1))
            labels.append(class_label)
    return torch.stack(data), np.array(labels)    

def prepare_data(video_folder, resize_dim=(224, 224)):
    finaldata=[]
    finallabel=[]
    data, labels=ExtractAlphabeticalDataset(video_folder,resize_dim,25)
    finaldata.extend(data)
    finallabel.extend(labels)
    # data, labels=ExtractCategoricalDataset(video_folder,resize_dim,10)
    # finaldata.append(data)
    # finallabel.append(labels)
    return torch.stack(finaldata), np.array(finallabel)

def main():
    cwd = os.getcwd()
    config_path = os.path.join(cwd,"PsychologyBot", "FrontEnd","Config", "config.yaml")
    config = yaml.safe_load(open(config_path))
    alphabetdataset=config['paths']['alphabetDataset']
    alphabet_dataset = cwd+ alphabetdataset
    data, labels = prepare_data(alphabet_dataset)

    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    # Save the splits
    torch.save(train_data, 'PsychologyBot/Frontend/Artifacts/ProcessedData/train_data_sign_prediction.pt')
    torch.save(test_data, 'PsychologyBot/Frontend/Artifacts/ProcessedData/test_data_sign_prediction.pt')
    np.save('PsychologyBot/Frontend/Artifacts/ProcessedData/train_labels_sign_prediction.npy', train_labels)
    np.save('PsychologyBot/Frontend/Artifacts/ProcessedData/test_labels_sign_prediction.npy', test_labels)

if __name__ == "__main__":
    main()