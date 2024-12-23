from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

def load_sequence_data(sequence_dir: Path) -> List[Dict[str, Any]]:
    """
    Load and validate sequence data from JSON files.
    
    Args:
        sequence_dir: Directory containing sequence files
        
    Returns:
        List of processed sequences with labels
    """
    processed_data = []
    gesture_counts = {}
    total_sequences = 0
    
    print(f"\nProcessing sequences from: {sequence_dir}")
    
    # Iterate through gesture folders
    for gesture_dir in sequence_dir.iterdir():
        if not gesture_dir.is_dir():
            continue
            
        gesture_label = gesture_dir.name
        gesture_counts[gesture_label] = 0
        
        # Process each sequence file
        for seq_file in gesture_dir.glob("*.json"):
            with seq_file.open() as f:
                sequence = json.load(f)
                
            # Validate sequence format
            if not sequence or not isinstance(sequence, list):
                print(f"Warning: Skipping invalid sequence: {seq_file}")
                continue
                
            processed_data.append({
                'label': gesture_label,
                'sequence': sequence,
                'length': len(sequence)
            })
            
            gesture_counts[gesture_label] += 1
            total_sequences += 1
            
            # Print progress
            if total_sequences % 10 == 0:
                print(f"Processed {total_sequences} sequences...")
    
    print(f"\nTotal sequences processed: {total_sequences}")
    for gesture, count in gesture_counts.items():
        print(f"Gesture '{gesture}': {count} sequences")
    
    return processed_data

def preprocess_sequences(sequences: List[Dict[str, Any]], save_path: Path) -> None:
    """
    Preprocess sequences for LSTM training.
    
    Args:
        sequences: List of loaded sequences
        save_path: Path to save processed data
    """
    processed_data = []
    
    for seq in sequences:
        # Extract features from landmarks
        features = []
        for frame in seq['sequence']:
            # Flatten landmarks to feature vector
            frame_features = []
            for landmark in frame:
                frame_features.extend([landmark['x'], landmark['y'], landmark['z']])
            features.append(frame_features)
            
        processed_data.append({
            'label': seq['label'],
            'features': features,
            'length': len(features)
        })
    
    # Save processed data
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w') as f:
        json.dump(processed_data, f)
    
    print(f"\nProcessed data saved to: {save_path}")

def main():
    base_dir = Path("data/annotations/sequences")
    save_path = Path("data/processed/sequences.json")
    
    sequences = load_sequence_data(base_dir)
    preprocess_sequences(sequences, save_path)

if __name__ == "__main__":
    main()