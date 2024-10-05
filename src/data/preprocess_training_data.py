from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple


def validate_text_file(file_path: Path) -> List[str]:
    """
    Validate the content of a text file.

    Args:
        file_path (Path): Path to the text file.

    Returns:
        List[str]: List of validated lines from the file.

    Raises:
        ValueError: If the file doesn't meet the expected format.
    """
    with file_path.open("r") as f:
        data = f.readlines()

    if len(data) != 21:
        raise ValueError(f"Expected 21 lines, got {len(data)}")

    for line_num, line in enumerate(data, 1):
        values = line.strip().split(",")
        if len(values) != 3:
            raise ValueError(f"Expected 3 values on line {line_num}, got {len(values)}")
    return data


def parse_float_values(data: List[str]) -> List[List[float]]:
    """
    Parse string data into lists of float values.

    Args:
        data (List[str]): List of strings to parse.

    Returns:
        List[List[float]]: Parsed float values.

    Raises:
        ValueError: If a non-float value is encountered.
    """
    parsed_data: List[List[float]] = []
    for line_num, line in enumerate(data, 1):
        values = line.strip().split(",")
        try:
            float_values = [float(x) for x in values]
            parsed_data.append(float_values)
        except ValueError:
            raise ValueError(
                f"Non-float value found on line {line_num}: {line.strip()}"
            )
    return parsed_data


def process_file(
    file_path: Path, gesture: str, output_dir: Path, total_count: int
) -> None:
    """
    Process a single file and save the result as JSON.

    Args:
        file_path (Path): Path to the input file.
        gesture (str): Gesture name.
        output_dir (Path): Directory to save the output.
        total_count (int): Current count of processed files.
    """
    validated_data = validate_text_file(file_path)
    parsed_data = parse_float_values(validated_data)
    data: Dict[str, List[List[float]]] = {gesture: parsed_data}
    output_filename = output_dir / f"{total_count}.json"
    save_as_json(data, output_filename)


def get_gestures(input_dir: Path) -> List[str]:
    """
    Get a list of gesture names from the input directory.

    Args:
        input_dir (Path): Path to the input directory.

    Returns:
        List[str]: List of gesture names.
    """
    return [d.name for d in input_dir.iterdir() if d.is_dir()]


def preprocess_data(input_dir: Path, output_dir: Path) -> Tuple[int, int, int]:
    """
    Preprocess data from the input directory and save to the output directory.

    Args:
        input_dir (Path): Path to the input directory.
        output_dir (Path): Path to the output directory.

    Returns:
        Tuple[int, int, int]: Number of gestures, invalid samples, and total samples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gestures = get_gestures(input_dir)
    print(f"Found {len(gestures)} gestures: {gestures}")

    invalid_count = 0
    total_count = 0

    for gesture in gestures:
        gesture_dir = input_dir / gesture
        for file_path in gesture_dir.glob("*.txt"):
            try:
                process_file(file_path, gesture, output_dir, total_count)
                total_count += 1
            except ValueError as e:
                print(f"Invalid sample: {file_path.name}. Error: {str(e)}")
                invalid_count += 1
    return len(gestures), invalid_count, total_count


def save_as_json(data: Dict[str, List[List[float]]], filename: Path) -> None:
    """
    Save data as a JSON file.

    Args:
        data (Dict[str, List[List[float]]]): Data to be saved.
        filename (Path): Path to the output JSON file.
    """
    with filename.open("w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    """
    Main function to run the preprocessing script.
    """
    input_dir = Path("data/annotations/text_files")
    output_dir = Path("data/train_data")

    num_gestures, invalid_samples, total_samples = preprocess_data(
        input_dir, output_dir
    )

    print(f"Processed data saved to {output_dir}")
    print(f"Number of gestures: {num_gestures}")
    print(f"Number of invalid samples: {invalid_samples}")
    print(f"Total samples processed: {total_samples}")
    print(
        f"Percentage of invalid samples: {invalid_samples / total_samples * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
