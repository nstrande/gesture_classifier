import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict
import shutil

class DataAugmenter:
    """
    Performs data augmentation on sign language sequences.
    
    Creates augmented versions of hand landmark sequences through:
    - Rotation: Small rotations of x,y coordinates
    - Noise: Adding small random variations to coordinates
    - Temporal scaling: Stretching/compressing sequences in time
    
    Args:
        input_dir: Directory containing original sequences
        output_dir: Directory to save augmented sequences
    """
    
    def __init__(self, input_dir: str = "data/annotations", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create clean output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def rotate_coordinates(self, sequence: List[List[Dict]], angle: float) -> List[List[Dict]]:
        """
        Rotates x,y coordinates by given angle.
        
        Args:
            sequence: List of frames, each containing landmark coordinates
            angle: Rotation angle in degrees
            
        Returns:
            Sequence with rotated coordinates
        """
        rotated_sequence = []
        
        # Convert angle to radians
        theta = np.radians(angle)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        for frame in sequence:
            rotated_frame = []
            for landmark in frame:
                point = np.array([landmark['x'], landmark['y']])
                rotated_point = np.dot(rot_matrix, point)
                
                rotated_frame.append({
                    'x': float(rotated_point[0]),
                    'y': float(rotated_point[1]),
                    'z': landmark['z']  # z coordinate remains unchanged
                })
            rotated_sequence.append(rotated_frame)
            
        return rotated_sequence

    def add_noise(self, sequence: List[List[Dict]], noise_level: float = 0.002) -> List[List[Dict]]:
        """
        Adds small random variations to coordinates.
        
        Args:
            sequence: List of frames containing landmark coordinates
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Sequence with added noise
        """
        noisy_sequence = []
        
        for frame in sequence:
            noisy_frame = []
            for landmark in frame:
                noisy_frame.append({
                    'x': landmark['x'] + np.random.normal(0, noise_level),
                    'y': landmark['y'] + np.random.normal(0, noise_level),
                    'z': landmark['z'] + np.random.normal(0, noise_level/2)  # less noise on z
                })
            noisy_sequence.append(noisy_frame)
            
        return noisy_sequence

    def temporal_scaling(self, sequence: List[List[Dict]], scale_factor: float) -> List[List[Dict]]:
        """
        Stretches or compresses sequence in time using linear interpolation.
        
        Args:
            sequence: List of frames containing landmark coordinates
            scale_factor: Factor to scale sequence length (e.g., 0.9 for 90% length)
            
        Returns:
            Time-scaled sequence with interpolated frames
        """
        num_frames = len(sequence)
        new_num_frames = int(num_frames * scale_factor)
        
        if new_num_frames <= 1:
            return sequence
            
        # Create interpolation points
        old_frames = np.arange(num_frames)
        new_frames = np.linspace(0, num_frames-1, new_num_frames)
        
        scaled_sequence = []
        # For each new frame
        for new_t in new_frames:
            new_frame = []
            
            # Find two nearest original frames
            i = int(np.floor(new_t))
            if i >= num_frames - 1:
                scaled_sequence.append(sequence[-1])
                continue
                
            # Linear interpolation between frames
            alpha = new_t - i
            for j in range(len(sequence[0])):  # For each landmark
                old_lm1 = sequence[i][j]
                old_lm2 = sequence[i+1][j]
                
                new_frame.append({
                    'x': old_lm1['x'] * (1-alpha) + old_lm2['x'] * alpha,
                    'y': old_lm1['y'] * (1-alpha) + old_lm2['y'] * alpha,
                    'z': old_lm1['z'] * (1-alpha) + old_lm2['z'] * alpha
                })
            
            scaled_sequence.append(new_frame)
            
        return scaled_sequence

    def augment_sequence(self, sequence: List[List[Dict]]) -> List[List[List[Dict]]]:
        """
        Generates multiple augmented versions of a sequence.
        
        Applies multiple augmentations:
        - Rotations: -10°, -5°, 5°, 10°
        - Noise: Two noise levels
        - Temporal scaling: 90% and 110% of original length
        
        Args:
            sequence: Original sequence of frames
            
        Returns:
            List of augmented sequences (including original)
        """
        augmented_sequences = []
        
        # Original sequence
        augmented_sequences.append(sequence)
        
        # Rotations
        angles = [-5, 5, -10, 10]
        for angle in angles:
            aug = self.rotate_coordinates(sequence, angle)
            augmented_sequences.append(aug)
        
        # Noise variations
        noise_levels = [0.002, 0.003]
        for noise in noise_levels:
            aug = self.add_noise(sequence, noise)
            augmented_sequences.append(aug)
        
        # Temporal scaling
        scales = [0.9, 1.1]
        for scale in scales:
            aug = self.temporal_scaling(sequence, scale)
            augmented_sequences.append(aug)
        
        return augmented_sequences

    def process_directory(self):
        """
        Processes all sequences in input directory.
        
        For each sequence:
        1. Copies original file to output directory
        2. Creates multiple augmented versions
        3. Saves augmented versions with suffix _aug_X
        """
        for gesture_dir in self.input_dir.iterdir():
            if not gesture_dir.is_dir():
                continue
                
            # Create output directory for this gesture
            output_gesture_dir = self.output_dir / gesture_dir.name
            output_gesture_dir.mkdir(exist_ok=True)
            
            # Copy original files first
            for seq_file in gesture_dir.glob("*.json"):
                shutil.copy2(seq_file, output_gesture_dir)
            
            # Create augmented versions
            for seq_file in gesture_dir.glob("*.json"):
                with open(seq_file) as f:
                    sequence = json.load(f)
                
                # Generate augmented versions
                augmented_sequences = self.augment_sequence(sequence)
                
                # Save each augmented sequence
                base_name = seq_file.stem
                for i, aug_seq in enumerate(augmented_sequences[1:], 1):  # Skip first (original)
                    aug_file = output_gesture_dir / f"{base_name}_aug_{i}.json"
                    with open(aug_file, 'w') as f:
                        json.dump(aug_seq, f)

def main():
    augmenter = DataAugmenter()
    augmenter.process_directory()
    
    # Print statistics - fix af file counting
    input_files = len(list(Path("data/annotations").glob("*/*.json")))
    output_files = len(list(Path("data/processed").glob("*/*.json")))
    
    print(f"Original files: {input_files}")
    print(f"After augmentation: {output_files}")
    print(f"Augmentation factor: {output_files/input_files:.1f}x")

if __name__ == "__main__":
    main()