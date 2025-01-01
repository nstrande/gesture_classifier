import json
import os
import shutil
from pathlib import Path

import numpy as np


def setup_directories(input_dir: str = "data/raw", output_dir: str = "data/processed"):
    """
    Sets up input and output directories for data processing.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create clean output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    return input_path, output_path


def mirror_coordinates(sequence: list[list[dict]]) -> list[list[dict]]:
    """
    Mirrors the x-coordinates to create a mirrored version of the sequence.
    This simulates signs performed with the opposite hand.

    Args:
        sequence: List of frames containing landmark coordinates

    Returns:
        Sequence with mirrored coordinates
    """
    mirrored_sequence = []

    for frame in sequence:
        mirrored_frame = []
        for landmark in frame:
            mirrored_frame.append(
                {
                    "x": -landmark["x"],  # Mirror the x-coordinate
                    "y": landmark["y"],  # y remains unchanged
                    "z": landmark["z"],  # z remains unchanged
                }
            )
        mirrored_sequence.append(mirrored_frame)

    return mirrored_sequence


def rotate_coordinates(sequence: list[list[dict]], angle: float) -> list[list[dict]]:
    """
    Rotates x,y coordinates by given angle.

    Args:
        sequence: List of frames containing landmark coordinates
        angle: Rotation angle in degrees
    """
    rotated_sequence = []

    theta = np.radians(angle)
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    for frame in sequence:
        rotated_frame = []
        for landmark in frame:
            point = np.array([landmark["x"], landmark["y"]])
            rotated_point = np.dot(rot_matrix, point)

            rotated_frame.append(
                {
                    "x": float(rotated_point[0]),
                    "y": float(rotated_point[1]),
                    "z": landmark["z"],
                }
            )
        rotated_sequence.append(rotated_frame)

    return rotated_sequence


def add_noise(
    sequence: list[list[dict]], noise_level: float = 0.002
) -> list[list[dict]]:
    """
    Adds small random variations to coordinates.

    Args:
        sequence: List of frames containing landmark coordinates
        noise_level: Standard deviation of Gaussian noise
    """
    noisy_sequence = []

    for frame in sequence:
        noisy_frame = []
        for landmark in frame:
            noisy_frame.append(
                {
                    "x": landmark["x"] + np.random.normal(0, noise_level),
                    "y": landmark["y"] + np.random.normal(0, noise_level),
                    "z": landmark["z"] + np.random.normal(0, noise_level / 2),
                }
            )
        noisy_sequence.append(noisy_frame)

    return noisy_sequence


def temporal_scaling(
    sequence: list[list[dict]], scale_factor: float
) -> list[list[dict]]:
    """
    Stretches or compresses sequence in time using linear interpolation.

    Args:
        sequence: List of frames containing landmark coordinates
        scale_factor: Factor to scale sequence length (e.g., 0.9 for 90% length)
    """
    num_frames = len(sequence)
    new_num_frames = int(num_frames * scale_factor)

    if new_num_frames <= 1:
        return sequence

    new_frames = np.linspace(0, num_frames - 1, new_num_frames)

    scaled_sequence = []
    for new_t in new_frames:
        new_frame = []

        i = int(np.floor(new_t))
        if i >= num_frames - 1:
            scaled_sequence.append(sequence[-1])
            continue

        alpha = new_t - i
        for j in range(len(sequence[0])):
            old_lm1 = sequence[i][j]
            old_lm2 = sequence[i + 1][j]

            new_frame.append(
                {
                    "x": old_lm1["x"] * (1 - alpha) + old_lm2["x"] * alpha,
                    "y": old_lm1["y"] * (1 - alpha) + old_lm2["y"] * alpha,
                    "z": old_lm1["z"] * (1 - alpha) + old_lm2["z"] * alpha,
                }
            )

        scaled_sequence.append(new_frame)

    return scaled_sequence


def augment_sequence(sequence: list[list[dict]]) -> list[list[list[dict]]]:
    """
    Generates multiple augmented versions of a sequence.
    Now including mirrored versions.

    Args:
        sequence: Original sequence of frames

    Returns:
        List of augmented sequences (including original)
    """
    augmented_sequences = []

    # Original sequence
    augmented_sequences.append(sequence)

    # Mirrored version
    mirrored = mirror_coordinates(sequence)
    augmented_sequences.append(mirrored)

    # Rotations
    angles = [-5, 5, -10, 10]
    for angle in angles:
        aug = rotate_coordinates(sequence, angle)
        augmented_sequences.append(aug)
        # Also add rotated versions of the mirrored sequence
        aug_mirrored = rotate_coordinates(mirrored, angle)
        augmented_sequences.append(aug_mirrored)

    # Noise variations
    noise_levels = [0.002, 0.003]
    for noise in noise_levels:
        aug = add_noise(sequence, noise)
        augmented_sequences.append(aug)
        # Add noise to mirrored version
        aug_mirrored = add_noise(mirrored, noise)
        augmented_sequences.append(aug_mirrored)

    # Temporal scaling
    scales = [0.9, 1.1]
    for scale in scales:
        aug = temporal_scaling(sequence, scale)
        augmented_sequences.append(aug)
        # Time scale the mirrored version as well
        aug_mirrored = temporal_scaling(mirrored, scale)
        augmented_sequences.append(aug_mirrored)

    return augmented_sequences


def normalize_sequence(sequence: list[list[dict]]) -> list[list[dict]]:
    """
    Normalizes an entire sequence relative to its global min/max values.
    This preserves relative movements between frames while scaling to [0,1] range.

    Args:
        sequence: List of frames containing landmark coordinates

    Returns:
        Normalized sequence with all coordinates in [0,1] range
    """
    # Find global min/max values across all frames
    x_values = [lm["x"] for frame in sequence for lm in frame]
    y_values = [lm["y"] for frame in sequence for lm in frame]
    z_values = [lm["z"] for frame in sequence for lm in frame]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    z_min, z_max = min(z_values), max(z_values)

    # Avoid division by zero - if min==max, keep original values
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    z_range = z_max - z_min if z_max != z_min else 1

    normalized_sequence = []
    for frame in sequence:
        normalized_frame = []
        for landmark in frame:
            normalized_frame.append(
                {
                    "x": (landmark["x"] - x_min) / x_range,
                    "y": (landmark["y"] - y_min) / y_range,
                    "z": (landmark["z"] - z_min) / z_range,
                }
            )
        normalized_sequence.append(normalized_frame)

    return normalized_sequence


def process_directory(input_path: Path, output_path: Path):
    """
    Processes all sequences in input directory.
    Now includes normalization as final step.
    """
    for gesture_dir in input_path.iterdir():
        if not gesture_dir.is_dir():
            continue

        output_gesture_dir = output_path / gesture_dir.name
        output_gesture_dir.mkdir(exist_ok=True)

        # Process each sequence file
        for seq_file in gesture_dir.glob("*.json"):
            with open(seq_file) as f:
                sequence = json.load(f)

            # First create augmented versions
            augmented_sequences = augment_sequence(sequence)

            # Save original and augmented sequences, applying normalization to each
            for i, aug_seq in enumerate(augmented_sequences):
                # Normalize the sequence
                normalized_seq = normalize_sequence(aug_seq)

                # Save normalized sequence
                if i == 0:
                    # Original sequence
                    output_file = output_gesture_dir / seq_file.name
                else:
                    # Augmented sequence
                    output_file = output_gesture_dir / f"{seq_file.stem}_aug_{i}.json"

                with open(output_file, "w") as f:
                    json.dump(normalized_seq, f)


def main():
    input_path, output_path = setup_directories()
    process_directory(input_path, output_path)

    input_files = len(list(Path("data/raw").glob("*/*.json")))
    output_files = len(list(Path("data/processed").glob("*/*.json")))

    print(f"Original files: {input_files}")
    print(f"After augmentation: {output_files}")
    print(f"Augmentation factor: {output_files/input_files:.1f}x")


if __name__ == "__main__":
    main()
