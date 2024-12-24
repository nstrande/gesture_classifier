from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from pathlib import Path


class JSONDataset(Dataset):
    def __init__(self, data_dir):
        self.sequences = []
        self.labels = []
        
        # Declaring the labels dictionary
        for gesture_dir in Path(data_dir).iterdir():
            if gesture_dir.is_dir():
                label = gesture_dir.name
                # Load alle sekvenser for dette tegn
                for seq_file in gesture_dir.glob("*.json"):
                    sequence = self.load_sequence(seq_file)
                    self.sequences.append(sequence)
                    self.labels.append(label)
        
        # Indexing the labels
        self.label_to_idx = {label: i for i, label in enumerate(sorted(set(self.labels)))}
        self.labels = [self.label_to_idx[label] for label in self.labels]
    
    def load_sequence(self, json_path):
        with open(json_path, 'r') as f:
            sequence = json.load(f)
        
        # Setting the min and max for each coordinate in the sequence
        all_x = [lm['x'] for frame in sequence for lm in frame]
        all_y = [lm['y'] for frame in sequence for lm in frame]
        all_z = [lm['z'] for frame in sequence for lm in frame]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)
        
        # Normalizing each frame
        frames = []
        for frame in sequence:
            landmarks = []
            for landmark in frame:
                landmarks.extend([
                    (landmark['x'] - min_x) / (max_x - min_x) if max_x != min_x else 0,
                    (landmark['y'] - min_y) / (max_y - min_y) if max_y != min_y else 0,
                    (landmark['z'] - min_z) / (max_z - min_z) if max_z != min_z else 0
                ])
            frames.append(landmarks)
            
        return torch.FloatTensor(frames)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
