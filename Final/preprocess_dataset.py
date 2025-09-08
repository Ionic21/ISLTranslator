#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Tuple, Optional
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

# Anchor to repo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset roots (ISL_CSLRT_Corpus layout)
SENTENCE_DIR = os.path.join(BASE_DIR, "ISL_CSLRT_Corpus", "Frames_Sentence_Level")
WORD_DIR     = os.path.join(BASE_DIR, "ISL_CSLRT_Corpus", "Frames_Word_Level")

# Image preprocessing (compact)
TARGET_SIZE = (32, 32)       # (W, H)
COLOR_MODE  = "grayscale"    # single channel to cut features by 3×
DTYPE       = np.float32
NORMALIZE   = True

# Sequence handling
MAX_SEQ_LEN = 40             # cap length to bound memory/compute
PAD_VALUE   = 0.0            # masked by zero

# Outputs
OUT_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(OUT_DIR, exist_ok=True)

# Sentence outputs (fixed-length 3D tensor: N x L x F)
SENT_X = os.path.join(OUT_DIR, "X_sentences.npy")
SENT_Y = os.path.join(OUT_DIR, "y_sentences.npy")
SENT_CLASSES = os.path.join(OUT_DIR, "sentence_labels.npy")

# Word outputs (2D features: N x F_word)
WORD_X = os.path.join(OUT_DIR, "X_words.npy")
WORD_Y = os.path.join(OUT_DIR, "y_words.npy")
WORD_CLASSES = os.path.join(OUT_DIR, "word_labels.npy")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _list_sorted_dirs(path: str) -> List[str]:
    try:
        items = os.listdir(path)
    except FileNotFoundError:
        return []
    return [d for d in sorted(items) if not d.startswith(".") and os.path.isdir(os.path.join(path, d))]

def _list_sorted_imgs(path: str) -> List[str]:
    try:
        items = os.listdir(path)
    except FileNotFoundError:
        return []
    return [f for f in sorted(items) if f.lower().endswith(VALID_EXTS)]

def _read_frame(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if COLOR_MODE == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype(DTYPE)
        if NORMALIZE:
            img /= 255.0
        img = np.expand_dims(img, axis=-1)  # (H,W,1) for consistent flatten
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype(DTYPE)
        if NORMALIZE:
            img /= 255.0
    return img

def _gather_two_level_sequences(root_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    sequences, labels = [], []
    for label_name in _list_sorted_dirs(root_dir):
        sentence_path = os.path.join(root_dir, label_name)
        rep_folders = _list_sorted_dirs(sentence_path)

        # If no repetition folders, treat the sentence folder as one sequence
        if not rep_folders:
            frames = []
            for f in _list_sorted_imgs(sentence_path):
                arr = _read_frame(os.path.join(sentence_path, f))
                if arr is not None:
                    frames.append(arr)
                if len(frames) >= MAX_SEQ_LEN:  # truncate online
                    break
            if frames:
                seq = np.stack(frames, axis=0)
                sequences.append(seq)
                labels.append(label_name)
            continue

        for rep in rep_folders:
            rep_path = os.path.join(sentence_path, rep)
            frames = []
            for f in _list_sorted_imgs(rep_path):
                arr = _read_frame(os.path.join(rep_path, f))
                if arr is not None:
                    frames.append(arr)
                if len(frames) >= MAX_SEQ_LEN:
                    break
            if frames:
                seq = np.stack(frames, axis=0)
                sequences.append(seq)
                labels.append(label_name)
    return sequences, labels

def _pad_truncate_time(seq: np.ndarray, L: int) -> np.ndarray:
    T = seq.shape[0]  # Get the time dimension (first axis)
    if T == L:
        return seq
    if T > L:
        return seq[:L]
    # pad
    pad_len = L - T
    pad_block = np.full((pad_len, *seq.shape[1:]), PAD_VALUE, dtype=seq.dtype)
    return np.concatenate([seq, pad_block], axis=0)

def build_sentence_level():
    print("Preprocessing sentence-level (fixed length, compact features)...")
    seqs, labs = _gather_two_level_sequences(SENTENCE_DIR)
    if not seqs:
        print("  Warning: no sentence sequences found.")
        np.save(SENT_X, np.empty((0, 0, 0), dtype=DTYPE))
        np.save(SENT_Y, np.empty((0,), dtype=np.int64))
        np.save(SENT_CLASSES, np.array([], dtype=str))
        return

    # Get frame dimensions from the first sequence
    H, W, C = seqs[0].shape[1:4]  # Fixed: access first element of list, then get shape
    F = H * W * C
    seqs_flat = [s.reshape(s.shape[0], -1).astype(np.float32) for s in seqs]  # Fixed: use s.shape[0] for time dim
    seqs_fixed = [_pad_truncate_time(s, MAX_SEQ_LEN) for s in seqs_flat]  # (L,F)
    X_sent = np.stack(seqs_fixed, axis=0)  # (N,L,F)

    # Encode labels
    le = LabelEncoder()
    y_sent = le.fit_transform(labs).astype(np.int64)

    np.save(SENT_X, X_sent)
    np.save(SENT_Y, y_sent)
    np.save(SENT_CLASSES, le.classes_)
    print(f"  Saved sentence tensors: {SENT_X}, {SENT_Y}, {SENT_CLASSES}")

def build_word_level():
    print("Preprocessing word-level (time pooling, compact features)...")
    if not os.path.isdir(WORD_DIR):
        print("  Skipped: WORD_DIR not found.")
        np.save(WORD_X, np.empty((0, 0), dtype=DTYPE))
        np.save(WORD_Y, np.empty((0,), dtype=np.int64))
        np.save(WORD_CLASSES, np.array([], dtype=str))
        return

    seqs, labs = _gather_two_level_sequences(WORD_DIR)
    if not seqs:
        print("  Warning: no word sequences found.")
        np.save(WORD_X, np.empty((0, 0), dtype=DTYPE))
        np.save(WORD_Y, np.empty((0,), dtype=np.int64))
        np.save(WORD_CLASSES, np.array([], dtype=str))
        return

    # Convert each sequence to a compact fixed-length vector via time pooling
    def to_fixed_vector(seq: np.ndarray) -> np.ndarray:
        # seq: (T,H,W,C) -> flatten per frame -> mean & std over time -> concat
        T = seq.shape[0]  # Fixed: get time dimension correctly
        flat = seq.reshape(T, -1).astype(np.float32)  # (T,F)
        m = flat.mean(axis=0)
        s = flat.std(axis=0)
        return np.concatenate([m, s]).astype(np.float32)  # (2F,)

    X2D = np.vstack([to_fixed_vector(s) for s in seqs]).astype(np.float32)

    le = LabelEncoder()
    y2d = le.fit_transform(labs).astype(np.int64)

    np.save(WORD_X, X2D)
    np.save(WORD_Y, y2d)
    np.save(WORD_CLASSES, le.classes_)
    print(f"  Saved word features: {WORD_X}, {WORD_Y}, {WORD_CLASSES}")

if __name__ == "__main__":
    # Reduce OpenCV threads for lighter CPU load
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    build_sentence_level()
    build_word_level()
    print("Done.")