#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import cv2
import mediapipe as mp
import yt_dlp
import hashlib
from pathlib import Path
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MSASL_TRAIN = "MSASL_train.json"
MSASL_VAL = "MSASL_val.json"
MSASL_CLASSES = "MSASL_classes.json"
VIDEO_DIR = os.path.join(BASE_DIR, "msasl_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

# Parameters
MAX_CLASSES = 10  # Focus on fewer classes for better accuracy
MIN_SAMPLES_PER_CLASS = 10  # Lower threshold for real data
MAX_SEQ_LEN = 30
MAX_TOTAL_SAMPLES = 200  # Reasonable limit for initial training

# Known problematic video IDs (add more as you discover them)
SKIP_VIDEO_IDS = {
    "1AyT77LqJzQ", "7y5Ye-2-ZBs", "AoQAPgEUIAs", "0FYsztUQnUM",
    "iMUjcZCLaGo", "e1q4pwMRTEc", "nI-XX3BtUlQ", "hEOcz-FHs4U",
    "ElL1zlHxo-0", "eIPe2WLrJtY", "rbnYbQyFvK4", "TzyuKvD8aXQ",
    "bNu8NnzOU-g", "Cgh1DXAQBuI", "7YYB3BEoksc", "73icFhednQU",
    "BC2ahgTLYbk", "xfVNJxKWqeg", "Kh8PEWtOZ6k", "Zh9LNvbksnI",
    "sN-PvjTdOCM", "uM98W6WkvK0", "XEdsQNNyp-E", "lSxAVvE9sPc",
    "Kg0VRsnehgQ", "h8dstbX5kN0", "3rurYLE33MI", "QxOYnMWCVhM",
    # Add more problematic IDs from your previous run
}

OUT_DIR = os.path.join(BASE_DIR, "processed_data_real_msasl")
os.makedirs(OUT_DIR, exist_ok=True)

def extract_mediapipe_features(frame):
    """Extract MediaPipe features from frame"""
    if frame is None or frame.size == 0:
        return None
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    
    features = []
    hands_detected = 0
    
    # Hand landmarks (up to 2 hands)
    if hand_results.multi_hand_landmarks:
        hands_detected = len(hand_results.multi_hand_landmarks)
        for hand_landmarks in hand_results.multi_hand_landmarks[:2]:
            hand_features = []
            for landmark in hand_landmarks.landmark:
                hand_features.extend([landmark.x, landmark.y, landmark.z])
            features.extend(hand_features)
    
    # Pad if less than 2 hands
    while len(features) < 21 * 3 * 2:  # 21 landmarks × 3 coords × 2 hands
        features.append(0.0)
    
    # Pose landmarks (upper body focus)
    if pose_results.pose_landmarks:
        # Use key upper body landmarks: shoulders, elbows, wrists
        upper_body_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        for idx in upper_body_indices:
            if idx < len(pose_results.pose_landmarks.landmark):
                landmark = pose_results.pose_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])
            else:
                features.extend([0.0, 0.0, 0.0])
    else:
        # Pad pose features
        features.extend([0.0] * (12 * 3))
    
    return np.array(features, dtype=np.float32) if features else None

def get_video_id_from_url(url):
    """Extract video ID from URL"""
    if 'youtube.com/watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    else:
        return hashlib.md5(url.encode()).hexdigest()[:10]

def check_video_accessibility(url):
    """Quick check if video might be accessible"""
    video_id = get_video_id_from_url(url)
    
    # Skip known problematic videos
    if video_id in SKIP_VIDEO_IDS:
        return False
        
    # Basic URL format check
    if 'youtube.com/watch?v=' not in url and 'youtu.be/' not in url:
        return False
        
    return True

def download_video_safe(url, video_id, timeout=30):
    """Download video with timeout and error handling"""
    output_path = os.path.join(VIDEO_DIR, f"{video_id}.%(ext)s")
    
    # Check if already downloaded
    for ext in ['mp4', 'webm', 'mkv']:
        filepath = os.path.join(VIDEO_DIR, f"{video_id}.{ext}")
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:  # At least 1KB
            return filepath
    
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'worst[height<=360]',  # Use lowest quality to save time/space
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': timeout,
        'retries': 1,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First check if info is accessible (quick test)
            info = ydl.extract_info(url, download=False)
            if info is None:
                # This catches cases where yt-dlp can't get basic info
                SKIP_VIDEO_IDS.add(video_id)
                return None
                
            # Attempt to download
            ydl.download([url])
            
            # --- NEW ROBUST CHECK ---
            downloaded_file = None
            for ext in ['mp4', 'webm', 'mkv', 'jpg']: # Add jpg in case of thumbnail
                filepath = os.path.join(VIDEO_DIR, f"{video_id}.{ext}")
                if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                    downloaded_file = filepath
                    break

            if downloaded_file:
                return downloaded_file
            else:
                # If no valid file was found, assume download failed silently
                SKIP_VIDEO_IDS.add(video_id)
                return None
            
    except Exception as e:
        # Add to skip list for future runs for any other error
        SKIP_VIDEO_IDS.add(video_id)
        return None
    
    return None

def extract_sequence_from_video(video_path, start_frame, end_frame):
    """Extract MediaPipe feature sequence from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    sequence = []
    target_frames = min(end_frame - start_frame + 1, MAX_SEQ_LEN)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_processed = 0
    for i in range(target_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract MediaPipe features
        features = extract_mediapipe_features(frame)
        if features is not None and len(features) == 162:  # Correct feature size
            sequence.append(features)
            frames_processed += 1
            
        # Stop if we have enough frames
        if frames_processed >= 20:  # At least 20 frames
            break
    
    cap.release()
    
    return np.array(sequence) if len(sequence) > 10 else None  # At least 10 frames

def load_and_filter_msasl_data():
    """Load MS-ASL data and filter accessible samples"""
    print("📖 Loading MS-ASL dataset...")
    
    try:
        with open(MSASL_CLASSES, 'r') as f:
            class_names = json.load(f)
            
        with open(MSASL_TRAIN, 'r') as f:
            train_data = json.load(f)
            
        with open(MSASL_VAL, 'r') as f:
            val_data = json.load(f)
            
        all_data = train_data + val_data
        print(f"Total samples in dataset: {len(all_data)}")
        
        # Filter accessible samples
        accessible_samples = []
        for sample in all_data:
            url = sample.get('url', '')
            if check_video_accessibility(url):
                accessible_samples.append(sample)
        
        print(f"Potentially accessible samples: {len(accessible_samples)}")
        return class_names, accessible_samples
        
    except FileNotFoundError as e:
        print(f"❌ MS-ASL files not found: {e}")
        return None, None

def select_target_classes(all_data, class_names):
    """Select classes with most accessible samples"""
    print(f"🎯 Selecting top {MAX_CLASSES} classes...")
    
    # Count samples per class
    class_counts = {}
    for sample in all_data:
        label = sample.get('label')
        if label is not None and label < len(class_names):
            class_name = class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Sort by frequency
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Select top classes with sufficient samples
    selected_classes = []
    for class_name, count in sorted_classes:
        if count >= MIN_SAMPLES_PER_CLASS and len(selected_classes) < MAX_CLASSES:
            selected_classes.append(class_name)
    
    print("Selected classes:")
    for i, class_name in enumerate(selected_classes):
        count = class_counts[class_name]
        print(f"  {i+1:2d}. {class_name}: {count} samples")
    
    return selected_classes

def process_real_msasl_samples(all_data, class_names, selected_classes):
    """Process real MS-ASL samples with MediaPipe"""
    print("\n🎬 Processing real MS-ASL videos with MediaPipe...")
    print("⏳ This may take several minutes...")
    
    # Create class mapping
    class_to_idx = {name: idx for idx, name in enumerate(selected_classes)}
    
    sequences = []
    labels = []
    processed = 0
    skipped = 0
    downloaded_videos = {}
    
    start_time = time.time()
    
    for i, sample in enumerate(all_data):
        # Stop if we have enough samples
        if processed >= MAX_TOTAL_SAMPLES:
            print(f"✋ Reached sample limit ({MAX_TOTAL_SAMPLES})")
            break
            
        label_idx = sample.get('label')
        if label_idx is None or label_idx >= len(class_names):
            continue
            
        class_name = class_names[label_idx]
        if class_name not in class_to_idx:
            continue
        
        url = sample.get('url', '')
        start_frame = sample.get('start', 0)
        end_frame = sample.get('end', 100)
        
        video_id = get_video_id_from_url(url)
        
        # Download video if not cached
        if video_id not in downloaded_videos:
            video_path = download_video_safe(url, video_id)
            downloaded_videos[video_id] = video_path
        else:
            video_path = downloaded_videos[video_id]
        
        if video_path is None:
            skipped += 1
            continue
        
        # Extract MediaPipe sequence
        sequence = extract_sequence_from_video(video_path, start_frame, end_frame)
        
        if sequence is not None:
            sequences.append(sequence)
            labels.append(class_to_idx[class_name])
            processed += 1
            
            # Progress update
            if processed % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  ✅ Processed: {processed} | ❌ Skipped: {skipped} | 📹 Videos: {len([v for v in downloaded_videos.values() if v])} | ⏱️ {elapsed:.1f}s")
        else:
            skipped += 1
    
    elapsed = time.time() - start_time
    success_rate = len([v for v in downloaded_videos.values() if v]) / len(downloaded_videos) * 100 if downloaded_videos else 0
    
    print(f"\n📊 Processing Complete:")
    print(f"✅ Successfully processed: {processed} sequences")
    print(f"❌ Skipped: {skipped} samples") 
    print(f"📹 Video success rate: {success_rate:.1f}%")
    print(f"⏱️ Total time: {elapsed/60:.1f} minutes")
    
    return sequences, labels, selected_classes

def save_real_mediapipe_data(sequences, labels, class_names):
    """Save processed real MS-ASL data"""
    if not sequences:
        print("❌ No sequences to save!")
        return
        
    print("💾 Saving real MS-ASL MediaPipe data...")
    
    # Pad sequences for sentence model
    max_len = max(len(seq) for seq in sequences)
    feature_dim = sequences[0].shape[1]  # 162
    
    X_sent = np.zeros((len(sequences), MAX_SEQ_LEN, feature_dim), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), MAX_SEQ_LEN)
        X_sent[i, :seq_len] = seq[:seq_len]
    
    y_sent = np.array(labels, dtype=np.int64)
    
    # Create word-level features
    X_word = []
    for seq in sequences:
        # Statistical features over time
        mean_feat = seq.mean(axis=0)
        std_feat = seq.std(axis=0)
        max_feat = seq.max(axis=0)
        min_feat = seq.min(axis=0)
        combined = np.concatenate([mean_feat, std_feat, max_feat, min_feat])
        X_word.append(combined)
    
    X_word = np.array(X_word)
    y_word = y_sent.copy()
    
    # Save all data
    np.save(os.path.join(OUT_DIR, "X_sentences.npy"), X_sent)
    np.save(os.path.join(OUT_DIR, "y_sentences.npy"), y_sent)
    np.save(os.path.join(OUT_DIR, "sentence_labels.npy"), np.array(class_names))
    
    np.save(os.path.join(OUT_DIR, "X_words.npy"), X_word)
    np.save(os.path.join(OUT_DIR, "y_words.npy"), y_word)
    np.save(os.path.join(OUT_DIR, "word_labels.npy"), np.array(class_names))
    
    print(f"✅ Real MS-ASL data saved:")
    print(f"   📝 Sentence data: {X_sent.shape}")
    print(f"   📝 Word data: {X_word.shape}")
    print(f"   🏷️ Classes: {class_names}")

def main():
    print("🎬 Real MS-ASL + MediaPipe Dataset Processor")
    print("=" * 50)
    
    # Load MS-ASL dataset
    class_names, all_data = load_and_filter_msasl_data()
    
    if not all_data:
        print("❌ No MS-ASL data available!")
        return
    
    # Select classes to work with
    selected_classes = select_target_classes(all_data, class_names)
    
    if not selected_classes:
        print("❌ No suitable classes found!")
        return
    
    # Process videos with MediaPipe
    sequences, labels, final_classes = process_real_msasl_samples(all_data, class_names, selected_classes)
    
    if not sequences:
        print("❌ No sequences extracted!")
        return
    
    # Save processed data
    save_real_mediapipe_data(sequences, labels, final_classes)
    
    print(f"\n🎉 Real MS-ASL processing complete!")
    print(f"📊 Dataset ready for training with {len(sequences)} real sign language sequences")
    print(f"\n📋 Next steps:")
    print(f"1. python train_real_models.py")
    print(f"2. python live_translation_real.py")

if __name__ == "__main__":
    main()
